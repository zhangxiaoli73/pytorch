#ifdef USE_C10D_NCCL

#include <exception>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/WaitCounter.h>
#include <c10/util/irange.h>
#include <c10/util/thread_name.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/NanCheck.hpp>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGCCL.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/torch.h>
#include <optional>

namespace c10d {

namespace {

// Get a key string from device
inline std::string getKeyFromDevice(at::Device& device) {
  return std::to_string(device.index());
}

inline at::DeviceIndex getIndexFromDeviceKey(const std::string& deviceKey) {
  // initialize the device index to -1, which is an invalid value.
  int index = -1;
  try {
    index = std::stoi(deviceKey);
  } catch (const std::invalid_argument& e) {
    LOG(ERROR) << c10::str(
        "Invalid deviceKey: ", deviceKey, ",", e.what(), ".");
  } catch (const std::out_of_range& e) {
    LOG(ERROR) << "Out of range: " << e.what();
  }
  return static_cast<at::DeviceIndex>(index);
}

std::string getKeySendRecv(int myRank, int peer) {
  int lowRank = myRank < peer ? myRank : peer;
  int highRank = myRank < peer ? peer : myRank;
  std::string sendRecvPair =
      std::to_string(lowRank) + ":" + std::to_string(highRank);
  return sendRecvPair;
}

// Get device from tensor
inline at::Device getDevice(at::Tensor& tensor) {
  return tensor.device();
}

void syncStream(
    at::Device& device,
    c10::Event& ncclEvent,
    c10::Stream& ncclStream) {
  ncclEvent.record(at::cuda::getCurrentCUDAStream(device.index()));
  ncclEvent.block(ncclStream);
}

} // namespace

// Map from each communicator to its device index.
// This map is used when register/deregister cache segments from cache
// allocator. See design notes below:
// - Each segment should be registered only to the communicator on the
//   same device.
// - We cannot reuse devNCCLCommMap_ in each ProcessGroup because the key may be
//   ranks rather than device in point-to-point case.
// - This map has also to be maintained as global variable since the register
//   hooks are called outside the scope of any PG, thus we need traverse
//   communicators in all PGs.
static std::unordered_map<std::shared_ptr<NCCLComm>, int> ncclCommDevIdxMap;
static std::mutex ncclCommDevIdxMapMutex;
static bool allocatorHooksAttached = false;

const int64_t ProcessGroupGCCL::kWatchdogThreadSleepMillis = 100;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;
thread_local uint64_t ProcessGroupGCCL::ncclActiveGroupCounter_ = 0;

std::ostream& operator<<(
    std::ostream& output,
    const ProcessGroupGCCL::WorkGCCL& WorkGCCL) {
  std::string workInfo;
  workInfo = c10::str(
      "WorkGCCL(",
      "SeqNum=",
      WorkGCCL.seq_,
      ", OpType=",
      opTypeToString(WorkGCCL.opType_),
      ", NumelIn=",
      WorkGCCL.numelIn_,
      ", NumelOut=",
      WorkGCCL.numelOut_,
      ", Timeout(ms)=",
      WorkGCCL.opTimeout_.count(),
      ")");
  return output << workInfo;
}

ProcessGroupGCCL::WorkGCCL::WorkGCCL(
    const std::string& pgUID,
    const std::string& pgDesc,
    at::Device& device,
    int rank,
    OpType opType,
    uint64_t seq,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputs,
    bool desyncDebug,
    bool enableTiming,
    bool cudaEventCacheEnabled,
    DebugLevel distDebugLevel)
    : Work(rank, opType, profilingTitle, inputs),
      pgUID_(pgUID),
      pgDesc_(pgDesc),
      device_(device),
      workStartTime_(std::chrono::steady_clock::now()),
      seq_(seq),
      timingEnabled_(enableTiming),
      distDebugLevel_(distDebugLevel) {
  // Creates the CUDA event wrappers
  // Note: The actual events are lazily created when first recorded to with
  // DEFAULT_FLAGS = cudaEventDisableTiming.
  if (cudaEventCacheEnabled) {
    ncclStartEvent_ = enableTiming
        ? ProcessGroupGCCL::CUDAEventCache::get().create(enableTiming)
        : nullptr;
    ncclEndEvent_ =
        ProcessGroupGCCL::CUDAEventCache::get().create(enableTiming);
  } else {
    ncclStartEvent_ = enableTiming
        ? std::make_shared<at::cuda::CUDAEvent>(cudaEventDefault)
        : nullptr;
    ncclEndEvent_ = std::make_shared<at::cuda::CUDAEvent>(
        enableTiming ? cudaEventDefault : cudaEventDisableTiming);
  }
}

ProcessGroupGCCL::WorkGCCL::WorkGCCL(const WorkGCCL& w)
    : Work(w.rank_, w.opType_),
      std::enable_shared_from_this<WorkGCCL>(w),
      pgUID_(w.pgUID_),
      pgDesc_(w.pgDesc_),
      device_(w.device_),
      ncclStartEvent_(w.ncclStartEvent_),
      ncclEndEvent_(w.ncclEndEvent_),
      ncclComm_(w.ncclComm_),
      blockingWait_(w.blockingWait_),
      opTimeout_(w.opTimeout_),
      ownedEphermeralTimeout_(w.ownedEphermeralTimeout_),
      workStartTime_(w.workStartTime_),
      seq_(w.seq_),
      startTraceUpdated_(w.startTraceUpdated_),
      numelIn_(w.numelIn_),
      numelOut_(w.numelOut_),
      store_(w.store_),
      timingEnabled_(w.timingEnabled_),
      trace_id_(w.trace_id_),
      distDebugLevel_(w.distDebugLevel_) {
  exception_ = w.exception_;
}

ProcessGroupGCCL::WorkGCCL::~WorkGCCL() = default;

bool ProcessGroupGCCL::WorkGCCL::isCompleted() {
  if (isAborted()) {
    checkAndSetException();
  }
  return exception() || finishedGPUExecutionInternal();
}


bool ProcessGroupGCCL::WorkGCCL::isSuccess() const {
  C10_THROW_ERROR(NotImplementedError, "Work::isSuccess() is deprecated");
}

void ProcessGroupGCCL::WorkGCCL::checkAndSetException() {
  if (exception()) {
    // We already have an exception.
    return;
  }

  auto exception_ptr = checkForNCCLErrors();
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
  if (exception_) {
    LOG(ERROR) << logPrefix() << "Collective " << *this
               << " raised the following async exception: "
               << getExceptionMsgFromExceptionPtr(exception_);
  }
}

const std::string& ProcessGroupGCCL::WorkGCCL::logPrefix() const {
  static std::string prefix = c10::str("[Rank ", rank_, "] ");
  return prefix;
}

void ProcessGroupGCCL::WorkGCCL::setException(
    std::exception_ptr exception_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
}

// Helper that checks if the NCCL kernels are completed on the GPUs
bool ProcessGroupGCCL::WorkGCCL::finishedGPUExecution() {
  checkAndSetException();
  return finishedGPUExecutionInternal();
}

bool ProcessGroupGCCL::WorkGCCL::startedGPUExecutionInternal() const {
  // if timing is disabled we won't have allocated start events
  if (!timingEnabled_) {
    return false;
  }
  // Checking the work's corresponding CUDA event's status
  if (!ncclStartEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupGCCL::WorkGCCL::finishedGPUExecutionInternal() const {
  // Checking the work's corresponding CUDA event's status
  // It calls `cudaEventQuery` eventually. Although this seems to be a
  // non-blocking call, but we did notice hangs in the past. It can
  // hang if another thread is holding the CUDA global context lock. For
  // example, when doing a `cudaDeviceSynchronize` or even
  // `cudaStreamSynchronize`.
  if (!ncclEndEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupGCCL::WorkGCCL::checkTimeout(
    std::optional<std::chrono::milliseconds> timeout) {
  STATIC_SCOPED_WAIT_COUNTER(
      pytorch.wait_counter.ProcessGroupGCCL__checkTimeout);
  auto currentTimepoint = std::chrono::steady_clock::now();
  auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTimepoint - workStartTime_);
  auto workTimeout = timeout ? *timeout : opTimeout_;

  if (timeElapsed < workTimeout)
    return false;

  // Timed out

  // There is already an error, we don't override it
  if (exception())
    return true;

  std::string exceptionMsg = c10::str(
      logPrefix(),
      "Watchdog caught collective operation timeout: ",
      *this,
      " ran for ",
      timeElapsed.count(),
      " milliseconds before timing out.");

  LOG(ERROR) << exceptionMsg;
  std::exception_ptr exception_ptr =
      std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exceptionMsg));
  setException(exception_ptr);
  return true;
}

void ProcessGroupGCCL::WorkGCCL::handleException(
    ErrorHandlingMode errorHandling) {
  if (exception_) {
    auto exceptionMsg = c10::str(
        "Some NCCL operations have failed or timed out. Due to the ",
        "asynchronous nature of CUDA kernels, subsequent GPU operations ",
        "might run on corrupted/incomplete data.");
    LOG(ERROR) << logPrefix() << exceptionMsg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupGCCL.WorkGCCL.handleException");

    if (SHOULD_TEAR_DOWN(errorHandling)) {
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process down.");
      LOG(ERROR) << logPrefix() << tearDownMsg;
      std::rethrow_exception(exception_);
    }
  }
}

void ProcessGroupGCCL::WorkGCCL::synchronize() {
  // Call Synchronize without a timeout. We use this method to avoid adding a
  // timeout argument to the public synchronize API.
  synchronizeInternal(kNoTimeout);
}

void ProcessGroupGCCL::WorkGCCL::synchronizeStream() {
  auto currentStream = at::cuda::getCurrentCUDAStream(device_.index());
  // Block the current stream on the NCCL stream
  ncclEndEvent_->block(currentStream);

  if (avoidRecordStreams_) {
    stashed_for_allocator_safety_->clear();
  }
}

// Waiting on the work's corresponding CUDA events
void ProcessGroupGCCL::WorkGCCL::synchronizeInternal(
    std::chrono::milliseconds timeout) {
  synchronizeStream();

  // In case of blocking, wait for the operation to complete.
  if (blockingWait_) {
    while (!isCompleted()) {
      bool timedOut = checkTimeout(
          timeout == kNoTimeout ? std::nullopt : std::make_optional(timeout));
      // Explicitly abort ncclComms here before throwing this timed out
      // exception to users.
      // If throwing timed out excepiton without aborting nccl communicators
      // here, it was observed that CUDA GPU will have 100% utilization and
      // can not run new events successfully.
      if (timedOut) {
        std::string exceptionMsg = c10::str(
            logPrefix(),
            "Work ",
            (*this),
            " timed out in blocking wait (TORCH_NCCL_BLOCKING_WAIT=1).");
        LOG(ERROR) << exceptionMsg;
        break;
      }
      // Yield
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
    // exception() includes timeout and error during blocking wait
    if (exception()) {
      // Abort NCCL communicators
      abort();
      // Throw exception (from main thread here)
      handleException(TearDown);
    }
  }

  // Device synchronize only after we've completed timeout checks.
  if (barrierTensor_.defined()) {
    // If we use the work to do barrier, we should block here
    // `dist.barrier()` only requires all CPU processes to enter this
    // function, hence we only need to make sure the dummy all-reduce has
    // completed. So we would only need to sync the **current stream** back to
    // host, and do not need to synchronize the entire device (which may have
    // kernels running on other streams).
    // Using `cudaStreamSynchronize` instead of `cudaDeviceSynchronize` can:
    // - lower chance of hang;
    // - CurrentCUDAStream is usually the context of the next operation in
    // Python, thus blocking current stream would already block the next
    // compute kernel;
    // - achieve better barrier performance.
    auto currentStream = at::cuda::getCurrentCUDAStream(device_.index());
    // CUDAStream wrapper will correctly use a DeviceGuard here
    currentStream.synchronize();
  }
}

// Same as calling synchronize().
bool ProcessGroupGCCL::WorkGCCL::wait(std::chrono::milliseconds timeout) {
  RECORD_PARAM_COMMS(
      static_cast<int>(this->seq_), // seq
      std::make_tuple(pgUID_, pgDesc_), // PG name tuple
      rank_, // rank
      "wait", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1,
      -1,
      static_cast<int>(1)); // number of device?
  synchronizeInternal(timeout);
  // TODO(kwen2501): this should be moved to c10d tests, to qualify a NCCL
  // upgrade. Once a NCCL version is qualified, this code should not be needed
  // at runtime.
#ifdef PGNCCL_ENABLE_HASH
  if (distDebugLevel_ >= DebugLevel::Detail) {
    auto numel = getTensorsNumel(*outputs_);
    auto hashValue = hashTensors(*outputs_);
    PRINT_COLLECTIVE_HASH_SIGNATURE(
        "output", opTypeToString(opType_), numel, hashValue);
  }
#endif
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupGCCL::WorkGCCL::abort() {
  // Abort all communicators of this work
  ncclComm_->ncclCommAbort();

  ncclCommDevIdxMapMutex.lock();
  ncclCommDevIdxMap.erase(ncclComm_);
  ncclCommDevIdxMapMutex.unlock();
}

static std::atomic<size_t> process_group_id = 0;

constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple. You are probably using multiple "
    "devices under one thread. The support for such usage has been deprecated. "
    "For details, please refer to "
    "https://pytorch.org/docs/stable/distributed.html#multi-gpu-collective-functions. "
    "ProcessGroupGCCL continues supporting multi-process and multi-thread modes.";

ProcessGroupGCCL::ProcessGroupGCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(options),
      ncclCommCounter_(0),
      traceKeyStart_(getTraceStartKey("NCCL", rank)),
      traceKeyEnd_(getTraceEndKey("NCCL", rank)),
      terminateProcessGroup_(false),
      collectiveDebugInfoMode_(false),
      local_id_(process_group_id++)) {
  init();
}


void ProcessGroupGCCL::setSequenceNumberForGroup() {
} // NCCL just starts sequence numbers at 0.

uint64_t ProcessGroupGCCL::getSequenceNumberForGroup() {
  return seqCollective_;
}

void ProcessGroupGCCL::enableCollectivesTiming() {
  enableTiming_.store(true);
}

void ProcessGroupGCCL::waitForFutureOrTimeout(
    std::future<bool>& fut,
    const std::chrono::milliseconds& timeOutMilSec,
    const std::string& futDescription,
    bool throwException,
    bool log) {
  std::string errorMsg;

  ::c10d::C10dLoggingData data;
  if (log) {
    data.integers["pg_id"] = local_id_;
    data.integers["rank"] = rank_;
    data.integers["global_rank"] = globalRank();
    data.strings["flight_recorder_version"] = c10d::version_val_str;
  }

  TORCH_CHECK(fut.valid(), "Expected a valid future");
  std::future_status status = fut.wait_for(timeOutMilSec);
  if (status == std::future_status::ready) {
    // Calling .get() will re-raise any exception from the future, and we don't
    // care about the retval
    try {
      bool result = fut.get();
      if (result) {
        LOG(INFO) << logPrefix()
                  << "future is successfully executed for: " << futDescription;
        if (log) {
          data.strings["status"] = "SUCCESS";
        }
      }
    } catch (const std::exception& e) {
      errorMsg = c10::str(
          logPrefix(),
          "Exception thrown when waiting for future ",
          futDescription,
          ": ",
          e.what());
      if (log) {
        data.strings["status"] = "EXCEPTION";
        data.strings["exception"] = e.what();
      }
      LOG(ERROR) << errorMsg;
    } catch (...) {
      errorMsg = c10::str(
          logPrefix(),
          "Unknown exception thrown when waiting for future ",
          futDescription);
      if (log) {
        data.strings["status"] = "EXCEPTION";
        data.strings["exception"] = "Unknown exception";
      }
      LOG(ERROR) << errorMsg;
    }
  } else {
    errorMsg = c10::str(
        logPrefix(),
        "Future for ",
        futDescription,
        " timed out after ",
        timeOutMilSec.count(),
        " ms");
    data.strings["status"] = "TIMEOUT";
    LOG(ERROR) << errorMsg;
  }
  if (log) {
    auto logger = c10d::C10dLogger::getLogger();
    if (logger) {
      logger->log(data);
    }
  }
  if (throwException && !errorMsg.empty()) {
    C10_THROW_ERROR(DistBackendError, errorMsg);
  }
}

void ProcessGroupGCCL::abortCommsFromMap(
    std::unordered_map<std::string, std::shared_ptr<NCCLComm>>& ncclCommsMap,
    std::optional<std::string> abortReason) {
  // The process may control multiple devices, loop through the communicators on
  // each device
  for (auto& it : ncclCommsMap) {
    auto& devName = it.first;
    auto& ncclComm = it.second;
    at::cuda::OptionalCUDAGuard gpuGuard;
    at::DeviceIndex deviceIndex = getIndexFromDeviceKey(devName);
    if (deviceIndex >= 0) {
      // For P2P comms, the deviceIndex could be -1 (invalid), as the keys in
      // the map could be non deviceIndex, but rank to rank numbers. So we
      // indeed need to check if deviceIndex >= 0
      // TODO: fix `getIndexFromDeviceKey` or fix `DeviceKey`
      gpuGuard.set_index(deviceIndex);
    }
    LOG(INFO) << logPrefix() << "ProcessGroupGCCL destroying ncclComm_ "
              << ncclComm->ncclComm_ << " on CUDA device: " << devName;
    ncclComm->ncclCommAbort(abortReason);
    // Note that we don't remove the aborted communicators from the
    // cache. The reason is that if we do remove the communicator
    // from the cache, it is possible that a new collective operation
    // calls `ncclCommInitRank` to create a new communicator whereas
    // other ranks might have failed/timed out and didn't enter
    // `ncclCommInitRank`. As a result, when there is a failure on
    // a communicator the application receives an exception and its
    // their responsibility to destroy the process group and recreate
    // it to recover from errors.

    LOG(INFO) << logPrefix() << "ProcessGroupGCCL destroyed "
              << " communicator on CUDA device: " << devName;
  }
}

// Abort all communicators on this rank
bool ProcessGroupGCCL::abort(std::optional<std::string> abortReason) {
  // This will log counter for how long the abort actually takes.
  STATIC_SCOPED_WAIT_COUNTER(pytorch.ProcessGroupGCCL__abort);
  // Remove record from global ncclCommDevIdxMapMutex before aboarting,
  // so that a new cache segment would not register to already aborded
  // communicators. Note that ncclCommDevIdxMap is a global container which may
  // contain other PG's communicators, thus we need to only erase communicators
  // for the current PG.
  ncclCommDevIdxMapMutex.lock();
  for (auto& it : devNCCLCommMap_) {
    auto& ncclComm = it.second;
    ncclCommDevIdxMap.erase(ncclComm);
  }
  ncclCommDevIdxMapMutex.unlock();

  std::lock_guard<std::mutex> lock(mutex_);
  abortCommsFromMap(devNCCLCommMap_, abortReason);
  abortCommsFromMap(inInitializationCommMap_, abortReason);
  return true;
}
// todo: do we need it?
void ProcessGroupGCCL::shutdown(std::optional<std::string> reason) {
  // Don't join threads here since the purpose of this method is to abort all
  // communicators and signal the threads to exit. Joining on the threads could
  // potentially block and hence avoid it in this method.
  terminateProcessGroup_.store(true);
  workMetaListCV_.notify_one();

  // lauch abort asynchrounously and wait for it to complete or timeout
  LOG(INFO) << logPrefix()
            << "Launching ProcessGroupGCCL abort asynchrounously.";
  std::future<bool> fut = std::async(
      std::launch::async, [this, &reason]() { return this->abort(reason); });

  waitForFutureOrTimeout(
      fut, options_->timeout, "ProcessGroup abort", true, false);
  LOG(INFO) << logPrefix() << "ProcessGroupGCCL aborts successfully.";

  // We need to wait for abort to finish before we can safely shut down
  // heartbeat monitoring thread.
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();
}

ProcessGroupGCCL::~ProcessGroupGCCL() {
  LOG(INFO) << logPrefix() << "ProcessGroupGCCL destructor entered.";

  if (!terminateProcessGroup_.load()) {
    if (rank_ % localDeviceCount_ == 0) {
      TORCH_WARN_ONCE(
          "WARNING: process group has NOT been destroyed before we destruct ProcessGroupGCCL. ",
          "On normal program exit, the application should call destroy_process_group to ",
          "ensure that any pending NCCL operations have finished in this process. "
          "In rare cases this process can exit before this point and block the progress of "
          "another member of the process group. This constraint has always been present, "
          " but this warning has only been added since PyTorch 2.4");
    }
    // If user haven't explicitly destroy/shutdown process group, destructor
    // needs to do so
    shutdown();
  }
}

bool ProcessGroupGCCL::dumpDebuggingInfo() {
  // Serialize all calls to this function to avoid corrupting data, but allow
  // multiple calls in one runtime. User is responsible for preserving the
  // output file from an earlier call before a later call overwrites it.
  static std::mutex writeDebugInfoMutex;
  std::lock_guard<std::mutex> lock(writeDebugInfoMutex);
  LOG(ERROR) << logPrefix() << "ProcessGroupGCCL preparing to dump debug info.";
  if (ncclTraceBufferSize_ > 0) {
    // We dump nccl trace into local disk by default and users can register
    // their customized writer by inheriting `DebugInfoWriter` via
    // `registerDebugInfoWriter`.
    auto ncclTrace = dump_nccl_trace(true, true, false);
    DebugInfoWriter& writer = DebugInfoWriter::getWriter(globalRank());
    LOG(INFO) << logPrefix() << "ProcessGroupGCCL dumping nccl trace to "
              << writer.getWriterTarget();
    writer.write(ncclTrace);
    return true;
  }
  return false;
}

// We want to have both PG ID and global unique ID (guid) for the logging
// prefix. PG ID records how many ProcessGroupGCCL objects were created on a
// specific rank and is a stable index across ranks, which lets users reason
// about, for example, the second PG we initialized on this rank is for FSDP,
// and corresponds with PG ID = 1 on other ranks as well. Unlike PG ID, guid (or
// group name) is a global unique ID across ranks. The guid is either a hash of
// all the ranks in the group or a counter of how many times
// `_process_group_name` is called, essentially it means how many times we
// have PGs users have created. Before using split_group, even if
// we are creating a new sub-PG, all ranks have to call the API at the same
// time, and this makes `group_name` a unique identifier for a group (PG).
std::string ProcessGroupGCCL::createLogPrefix() const {
  if (!pg_desc_.empty() && pg_desc_ != "undefined") {
    return c10::str(
        "[PG ID ",
        local_id_,
        " PG GUID ",
        pg_uid_,
        "(",
        pg_desc_,
        ") Rank ",
        rank_,
        "] ");
  }
  return c10::str(
      "[PG ID ", local_id_, " PG GUID ", pg_uid_, " Rank ", rank_, "] ");
}

const std::string& ProcessGroupGCCL::logPrefix() const {
  return logPrefix_;
}

const int& ProcessGroupGCCL::globalRank() const {
  static int globalRank = rank_;
  return globalRank;
}

const std::vector<uint64_t>& ProcessGroupGCCL::groupRanks() const {
  if (options_->global_ranks_in_group.empty() && local_id_ == 0) {
    static std::vector<uint64_t> globalRanks(size_);
    std::iota(globalRanks.begin(), globalRanks.end(), 0);
    return globalRanks;
  }
  return options_->global_ranks_in_group;
}

namespace {

// Check validity of tensor
void check_gpu_single_tensor(
    const at::Tensor& tensor,
    const bool p2p = false // whether operation is a P2P operation
) {
  if (!tensor.is_cuda() || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be CUDA and dense");
  }
  // Skip the following requirements for P2P operations
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    if (p2p) {
      TORCH_WARN_ONCE(
          "Detected non-contiguous tensor in P2P operations. It is user "
          "responsibility to guarantee that source and destination tensors have "
          "the same contiguity format.");
    } else {
      C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
    }
  }
}

// Checks that all `tensors' have the same type and shape and reside on the same
// GPU.
// TODO: test_c10d_nccl.py should consider adding tests for the error conditions
// here, ie, that deliberately pass invalid tensors and check the right
// exception is thrown. The "Expected list of tensors on the same device"
// condition may be a challenge because the test would need to pass tensors on
// different devices in the same process.
int64_t check_gpu_tensors_same_device(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    C10_THROW_ERROR(ValueError, "Tensor list must be nonempty");
  }

  const auto& first = tensors.front();

  int64_t total_numel = 0;
  for (const auto& t : tensors) {
    if (!t.is_cuda() || t.is_sparse()) {
      C10_THROW_ERROR(ValueError, "Tensors must be CUDA and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      C10_THROW_ERROR(TypeError, "Tensors must have identical type");
    }
    if (!t.is_non_overlapping_and_dense()) {
      C10_THROW_ERROR(ValueError, "Tensors must be non-overlapping and dense");
    }
    // If we're in this function, the user called a _coalesced collective
    // on a set of tensors with potentially different sizes and strides.
    // Therefore, we don't check for matching sizes and strides,
    // but we do double-check tensors are on the same device.
    TORCH_CHECK_WITH(
        ValueError,
        t.get_device() == tensors[0].get_device(),
        "Expected list of tensors on the same device");
    total_numel += t.numel();
  }

  return total_numel;
}

bool check_same_size(const std::vector<at::Tensor>& input_tensors) {
  for (const auto& input_tensor : input_tensors) {
    if (!input_tensors[0].is_same_size(input_tensor)) {
      return false;
    }
  }
  return true;
}

} // namespace

c10::intrusive_ptr<ProcessGroupGCCL::WorkGCCL> ProcessGroupGCCL::initWork(
    at::Device& device,
    int rank,
    OpType opType,
    const char* profilingTitle,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs, // TODO(kwen2501): necessary?
    bool record) {
  auto r = c10::make_intrusive<ProcessGroupGCCL::WorkGCCL>(
      pg_uid_,
      pg_desc_,
      device,
      rank,
      opType,
      seqCollective_,
      profilingTitle,
      profilingTitle != nullptr ? std::optional<std::vector<at::Tensor>>(inputs)
                                : std::nullopt,
      desyncDebug_,
      enableTiming_.load(),
      cudaEventCacheEnabled_.load(),
      dist_debug_level_);
  return r;
}

// TODO(kwen2501): deprecate
std::vector<at::Tensor> ProcessGroupGCCL::WorkGCCL::result() {
  return *outputs_;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupGCCL::WorkGCCL::
    getFuture() {
  return future_;
}

float ProcessGroupGCCL::WorkGCCL::getDuration() const {
  TORCH_CHECK(timingEnabled_, "getDuration only works if timing was enabled");
  TORCH_CHECK(
      ncclStartEvent_,
      "getDuration only works if ncclStartEvents_ is populated, true if timing enabled");
  TORCH_CHECK(
      ncclEndEvent_,
      "getDuration only works if ncclEndEvents_ is populated, which should always be true");
  return ncclStartEvent_->elapsed_time(*ncclEndEvent_);
}

uint64_t ProcessGroupGCCL::WorkGCCL::getSequencenumber() const {
  return seq_;
}

void ProcessGroupGCCL::assignTimeoutToWork(
    const c10::intrusive_ptr<ProcessGroupGCCL::WorkGCCL>& work,
    const c10::intrusive_ptr<ProcessGroupGCCL::Options>& option) {
  std::chrono::milliseconds timeout = option->timeout;
  std::lock_guard<std::mutex> timeoutLock(mtxTimeoutExtension_);
  if (ephemeralTimeoutActive_.count() > 0) {
    timeout += ephemeralTimeoutActive_;
  }
  work->opTimeout_ = timeout;
}

void ProcessGroupGCCL::workEnqueue(
    c10::intrusive_ptr<ProcessGroupGCCL::WorkGCCL> work) {
  if (!terminateProcessGroup_.load()) {
    std::lock_guard<std::mutex> lock(workMetaListMutex_);
    // Avoid view tensors to be processed in cleanup thread.
    // View tensors' destruction invokes autograd_meta, which
    // needs to be destructed in user thread. Otherwise will
    // get deadlock. Here we enqueue work without outputs_.
    workMetaList_.emplace_back(*work);
    // update the PG status related to the last enqueued work
    pgStatus_->lastEnqueuedSeq = work->seq_;
    pgStatus_->lastEnqueuedWorkName = opTypeToString(work->opType_);
    pgStatus_->lastEnqueuedNumelIn = work->numelIn_;
    pgStatus_->lastEnqueuedNumelOut = work->numelOut_;
    lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
  }
}

ProcessGroupGCCL::Options::Options(bool is_high_priority_stream)
    : Backend::Options(NCCL_BACKEND_NAME, kProcessGroupGCCLDefaultTimeout),
      is_high_priority_stream(is_high_priority_stream) {}

static constexpr int CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;

void ProcessGroupGCCL::startCoalescing() {
  // Other collective ops bump seq_ before creating a work. Thus, if coalesced
  // ops bump seq_ only after initing a work they will collide with (reuse) the
  // seq_ of the last non-coalesced collective.  Previously, seq_ was bumped
  // inside endCoalescing, but before initWork. Since we now record individual
  // ops from a coalesce group into the flight recorder, we want to have the
  // same seq_ for those ops and its 'endCoalescing' op. Hence we bump during
  // start, which has one minor downside- we burn a seq_ if someone ever does a
  // 'start' and 'end' coalescing region without doing an operation inbetween.

  // Don't bump op_id_ here, because startCoalescing isn't a logical operation.
  // Bump it for each logical op inside the coalescing group.
  if (coalescing_state_ & CoalP2P) {
    seqP2P_++;
  } else {
    seqCollective_++;
  }

  coalescedDevice_.set_index(-1);
  coalescedComm_ = nullptr;
  coalescing_state_ |= CoalActive;
  groupStart();
}

// `optype` is for specifying a composite optype, such as ALLGATHER and
// REDUCE_SCATTER
c10::intrusive_ptr<Work> ProcessGroupGCCL::endCoalescing(OpType optype) {
  if (coalescedComm_ == nullptr) {
    // There is no actual work being coalesced, return here
    groupEnd();
    coalescing_state_ = 0;
    return nullptr;
  }
  TORCH_CHECK(
      coalescedDevice_.index() >= 0,
      "Somthing went wrong. Did you call end_coalescing before start_coalescing?");

  // `coalescedComm_` should have same set of comms across collectives
  auto comm = coalescedComm_;
  // `coalescedDevice_` should have same set of devices across collectives
  auto device = coalescedDevice_;

  // `getKeyFromDevice` is how we get keys for both collectives and batch P2P
  const auto key = getKeyFromDevice(device);
  auto ncclStream = ncclStreams_.at(key);

  // Create Work object
  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();
  bool enqueue =
      (coalescing_state_) && capture_status == c10::cuda::CaptureStatus::None;
  auto work =
      initWork(device, rank_, optype, "nccl:coalesced", {}, {}, enqueue);
  work->ncclComm_ = comm;
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams_;
  work->store_ = store_;
  assignTimeoutToWork(work, options_);

  // Record start before ncclGroupEnd
  if (work->timingEnabled_) {
    work->ncclStartEvent_->record(ncclStream);
  }

  groupEnd();

  // Record end after ncclGroupEnd
  // TODO(eqy): is this still necessary if avoidRecordStreams_ is set?
  work->ncclEndEvent_->record(ncclStream);

  if (avoidRecordStreams_) {
    // other functions expect an initialized ptr if avoidRecordStreams_ is set
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>();
  }

  // Notify graphs before we check the capture status preemptively
  at::cuda::CUDAGraph::inc_pending_event_queries();

  if (enqueue) {
    workEnqueue(work);
  } else {
    at::cuda::CUDAGraph::dec_pending_event_queries();
  }

  coalescing_state_ = 0;
  coalescedComm_ = nullptr;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupGCCL::endCoalescing() {
  // Default OpType to COALESCED if not specified
  return endCoalescing(OpType::COALESCED);
}

c10::intrusive_ptr<Work> ProcessGroupGCCL::allreduce_sparse(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  raise exception("not supported");
}

std::shared_ptr<NCCLComm> communicatorImpl() {
    #ifdef NCCL_HAS_COMM_NONBLOCKING
    ncclComm = NCCLComm::create(numRanks, rank, ncclID, options_->config);
    #else
    ncclComm = NCCLComm::create(numRanks, rank, ncclID);
    #endif
    return ncclComm;
}

c10::intrusive_ptr<Work> ProcessGroupGCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    TORCH_CHECK(
        complexViewAsRealAllowed(opts.reduceOp),
        "all_reduce does not support",
        opts.reduceOp,
        "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  check_gpu_single_tensor(tensor);

  if (intraNodeComm_ != nullptr && opts.reduceOp == ReduceOp::SUM) {
    using namespace intra_node_comm;
    auto algo = intraNodeComm_->selectAllReduceAlgo(tensor);
    if (algo != intra_node_comm::AllReduceAlgo::NONE) {
      intraNodeComm_->allReduce(tensor, algo);
      return c10::make_intrusive<IntraNodeCommWork>();
    }
  }
  TORCH_CHECK(
      !isFloat8Type(tensor.scalar_type()),
      "Float8 dtypes are not currenlty supported for NCCL reductions");
  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return collective(
      tensor,
      tensor,
      [&](allreduce_impl(at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream),
      OpType::ALLREDUCE,
      "nccl:all_reduce");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  auto total_numel = check_gpu_tensors_same_device(tensors);
  TORCH_CHECK(
      !isFloat8Type(tensors.back().scalar_type()),
      "Float8 dtypes are not currenlty supported for NCCL reductions");

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce_coalesced", // collective name
      total_numel, // inNelems
      total_numel, // outNelems
      tensors[0].scalar_type(), // dType
      // I'm not sure what in,outSplitSizes mean here.
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return collectiveCoalesced(
      tensors,
      tensors,
      [&](allreduce_impl(at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream),
      OpType::COALESCED,
      "nccl:allreduce_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    tensor = at::view_as_real(tensor);
  }
  check_gpu_single_tensor(tensor);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      opts.rootRank, // root rank
      "broadcast", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  const auto root = opts.rootRank + opts.rootTensor;
  bool nanCheck = (root == rank_);

  return collective(
      tensor,
      tensor,
      [&](broadcast_impl(at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream),
      OpType::BROADCAST,
      "nccl:broadcast",
      avoidRecordStreams,
      nanCheck);
}

// _broadcast_oop adds an out-of-place broadcast in PGNCCL
// Custom collectives may be implemented by coalescing broadcast operations
// One use-case is implementing a vector all_gather (all_gather_v)
// where unevenly sized inputs are gathered among participating ranks
// Since all_gather provides an out-of-place API, an all_gather_v
// semantic implemented inside pg_nccl.all_gather also needs to support
// out-of-place, for which an out-of-place broadcast is required to be added
c10::intrusive_ptr<Work> ProcessGroupNCCL::_broadcast_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const BroadcastOptions& opts) {
  if (outputTensor.numel() != inputTensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _broadcast_oop must have the same number of elements ");
  }
  const auto root = opts.rootRank + opts.rootTensor;
  bool nanCheck = (root == rank_);
  return collective(
      inputTensor,
      outputTensor,
      [&](broadcast_impl(at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream),
      OpType::BROADCAST,
      "nccl:_broadcast_oop",
      /*avoidRecordStreams=*/false,
      nanCheck);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    TORCH_CHECK(
        complexViewAsRealAllowed(opts.reduceOp),
        "reduce does not support",
        opts.reduceOp,
        "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  check_gpu_single_tensor(tensor);
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      opts.rootRank, // root rank
      "reduce", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return collective(
      tensor,
      tensor,
      [&](reduce_impl(at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream),
      OpType::REDUCE,
      "nccl:reduce");
}

// _reduce_oop exposes an out-of-place reduce from PGNCCL
// Custom collectives may be implemented by coalescing reduce operations
// One use-case is implementing a vector reduce_scatter (reduce_scatter_v)
// where inputs are reduced and scattered unevenly among participating ranks
// Since reduce_scatter provides an out-of-place API, a reduce_scatter_v
// semantic implemented inside pg_nccl.reduce_scatter also needs to support
// out-of-place, for which an out-of-place reduce is required to be added
c10::intrusive_ptr<Work> ProcessGroupNCCL::_reduce_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceOptions& opts) {
  if (outputTensor.numel() != inputTensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _reduce_oop must have the same number of elements ");
  }
  return collective(
      inputTensor,
      outputTensor,
      [&](reduce_impl(at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream),
      OpType::REDUCE,
      "nccl:_reduce_oop");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto inputTensor = inputTensors.back();
  check_gpu_single_tensor(inputTensor);
  // @lint-ignore CLANGTIDY
  auto outputTensors_ = outputTensors.back();

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "all_gather", // collective name
      inputTensor.numel(), // inNelems
      inputTensor.numel() * // outNelems
          this->getSize(),
      inputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  bool same_size = check_same_size(outputTensors_);
  if (same_size) {
    // Flatten a vector of tensors into a single, stacked tensor.
    at::Tensor outputFlattened = newLikeFlat(outputTensors_);

    return collective(
        inputTensor,
        outputFlattened,
        [&](allgather_impl(at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream),
        [](at::cuda::CUDAStream& ncclStream,
           c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          // avoidRecordStreams_ note: We actually don't need to stash anything
          // here.
          //  - inputTensors is stashed onto work->stashed_for_allocator_safety_
          //    in collective().
          //  - outputFlattened is stashed onto work->outputs_ in collective().
          //  - User-facing outputTensors should be held by the user until after
          //    waiting on work_, or the call makes no sense.
          // So all participating tensors are accounted for, and won't be
          // released back to their allocation streams until after work_ is
          // waited on.
                if (!avoidRecordStreams_) {
        c10::cuda::CUDACachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
      }
        },
        [&](at::cuda::CUDAStream& ncclStream,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          // Copy the flattened output tensors to the outputs.
          at::cuda::CUDAStreamGuard guard(ncclStream);
          for (const auto j : c10::irange(outputTensors_.size())) {
            // See [Sync Streams].
            if (!avoidRecordStreams_) {
              c10::cuda::CUDACachingAllocator::recordStream(
                  outputTensors_[j].storage().data_ptr(), ncclStream);
            }
            outputTensors_[j].copy_(outputFlattened[j], true);
          }
        },
        OpType::ALLGATHER,
        "nccl:all_gather");
  } else {
    const auto num_reduces = outputTensors_.size();
    startCoalescing();
    for (const int i : c10::irange(num_reduces)) {
      auto& output = outputTensors_[i];
      auto& input = (i == rank_) ? inputTensor : output;
      auto broadcastOpts = BroadcastOptions{
          static_cast<int64_t>(i), static_cast<int64_t>(0), opts.timeout};
      _broadcast_oop(output, input, broadcastOpts);
    }
    auto work = endCoalescing(OpType::ALLGATHER);
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  C10_THROW_ERROR(
      NotImplementedError,
      "ProcessGroupNCCL does not support allgather_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  return collectiveCoalesced(
      inputs,
      outputs,
      [&](allgather(at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream),
      OpType::COALESCED,
      "nccl:all_gather_into_tensor_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto outputTensor = outputTensors.back();
  check_gpu_single_tensor(outputTensor);
  // @lint-ignore CLANGTIDY
  auto inputTensors_ = inputTensors.back();
  TORCH_CHECK(
      !isFloat8Type(outputTensor.scalar_type()),
      "Float8 dtypes are not currenlty supported for NCCL reductions");

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "reduce_scatter", // collective name
      outputTensor.numel() * this->getSize(), // inNelems
      outputTensor.numel(), // outNelems
      outputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  bool same_size = check_same_size(inputTensors_);
  if (same_size) {
    // Flatten a vector of tensors into a single, stacked tensor.
    at::Tensor inputFlattened = newLikeFlat(inputTensors_);

    return collective(
        inputFlattened,
        outputTensor,
        [&](reduce_scatter_impl(at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream),
        [&](at::cuda::CUDAStream& ncclStream,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          if (avoidRecordStreams_) {
            // We only need to stash inputTensors.
            //  - inputFlattened is stashed onto
            //  work->stashed_for_allocator_safety_
            //    in collective().
            //  - User-facing outputTensors is stashed onto work->outputs_ in
            //  collective(),
            //    and should also be held by the user until after waiting on
            //    work_.
            auto& v = work->stashed_for_allocator_safety_;
            v->insert(v->end(), inputTensors_.begin(), inputTensors_.end());
          }

          // Copy the input tensors to the flattened inputs.
          at::cuda::CUDAStreamGuard guard(ncclStream);
          for (const auto j : c10::irange(inputTensors_.size())) {
            // See [Sync Streams].
            if (!avoidRecordStreams_) {
              c10::cuda::CUDACachingAllocator::recordStream(
                  inputTensors_[j].storage().data_ptr(), ncclStream);
            }
            inputFlattened[j].copy_(inputTensors_[j], true);
          }
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
        },
        [&](at::cuda::CUDAStream&,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
        OpType::REDUCE_SCATTER,
        "nccl:reduce_scatter");
  } else {
    const auto num_reduces = inputTensors_.size();
    startCoalescing();
    for (const int i : c10::irange(num_reduces)) {
      auto& input = inputTensors_[i];
      auto& output = (i == rank_) ? outputTensor : input;
      auto reduceOpts = ReduceOptions{
          opts.reduceOp,
          static_cast<int64_t>(i),
          static_cast<int64_t>(0),
          opts.timeout};
      _reduce_oop(output, input, reduceOpts);
    }
    auto work = endCoalescing(OpType::REDUCE_SCATTER);
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  if (inputTensor.dtype() != outputTensor.dtype()) {
    C10_THROW_ERROR(
        TypeError, "input tensor must be the same type as the output tensor.");
  }

  if (inputTensor.numel() != outputTensor.numel() * size_) {
    C10_THROW_ERROR(
        ValueError,
        "input tensor must be the same size as output size times world size");
  }

  // @lint-ignore CLANGTIDY
  const auto& tensor = outputTensor;
  TORCH_CHECK(
      !isFloat8Type(tensor.scalar_type()),
      "Float8 dtypes are not currenlty supported for NCCL reductions");
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensor, // inputTensor
      outputTensor, // outputTensor
      rank_, // rank
      "_reduce_scatter_base", // collective name
      inputTensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dtype
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash inputs and outputs.
  // Note 2: for asyncOp = false, we don't want to record streams because we
  // know that the NCCL stream will join back to the "current" stream right
  // after this op. So we might just as well keep the stream ownership of the
  // input/output tensors unchanged. The benefit would be that the
  // allocation/free of the tensors would look deterministic to the "current"
  // stream so that the caching allocator can reuse memory pool for this stream
  // in a clever way. This setting is added for libraries like FSDP which uses
  // `reduce_scatter_tensor`.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      inputTensor,
      outputTensor,
      [&](reduce_scatter_impl(at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream),
      [&]{
        if (!avoidRecordStreams) {
          c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
      },
      OpType::_REDUCE_SCATTER_BASE,
      "nccl:_reduce_scatter_base",
      avoidRecordStreams);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(
      !isFloat8Type(inputs.back().scalar_type()),
      "Float8 dtypes are not currenlty supported for NCCL reductions");
  return collectiveCoalesced(
      inputs,
      outputs,
      [&](reduce_scatter_impl(at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream),
      [&]{
        if (!avoidRecordStreams) {
          c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
      },
      OpType::COALESCED,
      "nccl:reduce_scatter_tensor_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::barrier(const BarrierOptions& opts) {
  RECORD_PARAM_COMMS(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      rank_, // rank
      "barrier", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // Device to use for barrier
  int barDevIdx = -1;

  // Select device to use for barrier
  // 1st choice: Use user defined GPU device ids if provided
  if (!opts.device_ids.empty()) {
    // Use the first device id because PG NCCL is single-device now
    barDevIdx = opts.device_ids[0];
  } else if (getBoundDeviceId()) {
    // 2nd choice: Use the bound GPU device id if available.
    // Bounded device id can be passed to `init_process_group`.
    barDevIdx = (*getBoundDeviceId()).index();
  } else if (!usedDeviceIdxs_.empty()) {
    // 3rd choice: infer the device id from the used device ids.
    barDevIdx = *usedDeviceIdxs_.begin();
  } else {
    // This means there is not yet a NCCL collective being called
    // Here we have to use the best guesses and will use a single GPU to call
    // allreduce to achieve barrier.
    // In case the multiple processes fall into the same node, we use rank to
    // ensure that each process is on a different GPU
    // Note: it is better to use global rank because the group-local rank can be
    // offset wrt the device id if intra-node GPUs are sharded into multiple
    // dimensions.
    barDevIdx = static_cast<int16_t>(globalRank() % localDeviceCount_);
    LOG(WARNING)
        << logPrefix()
        << c10::str(
               " using GPU ",
               barDevIdx,
               " to perform barrier as devices used by this process are currently unknown. ",
               "This can potentially cause a hang if this rank to GPU mapping is incorrect.",
               "Specify device_ids in barrier() to force use of a particular device,",
               "or call init_process_group() with a device_id.");
  }

  TORCH_CHECK_WITH(
      ValueError,
      barDevIdx >= 0,
      "Failed to infer a GPU device id to perform barrier. ");
  auto barDevice = at::Device(at::DeviceType::CUDA, barDevIdx);

  // Create a dummy tensor on the device
  // Note: we use zeros() instead of empty() to prevent barrier from triggering
  // alarm when NaN checker is enabled.
  at::Tensor barrierTensor =
      at::zeros({1}, at::TensorOptions().device(barDevice).dtype(at::kFloat));

  // All reduce to achieve the barrier
  auto work = allreduce(barrierTensor);

  // Work will take over barrierTensors
  auto ncclWork = dynamic_cast<ProcessGroupNCCL::WorkNCCL*>(work.get());
  TORCH_CHECK(ncclWork);
  ncclWork->barrierTensor_ = std::move(barrierTensor);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  check_gpu_single_tensor(outputTensor, true);
  check_gpu_single_tensor(inputTensor, true);
  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_all", // collective name
        inputTensor.numel(), // inNelems
        outputTensor.numel(), // outNelems
        inputTensor.scalar_type(), // dType
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>(), // outSplitSizes
        globalRankStart, // globalRankStart
        globalRankStride, // globalRankStride
        this->getSize()); // worldSize

    // avoidRecordStreams_ note: collective() will stash inputTensors and
    // outputTensors.
    return collective(
        inputTensor,
        outputTensor,
        [&](all2all_impl(at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream),
        [&]{
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
        }
        OpType::ALLTOALL_BASE,
        "nccl:all_to_all");
  } else {
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);

    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_allv", // collective name
        inputTensor.numel(), // inNelems
        outputTensor.numel(), // outNelems
        inputTensor.scalar_type(), // dType
        inputSplitSizes, // inSplitSizes
        outputSplitSizes, // outSplitSizes
        globalRankStart, // globalRankStart
        globalRankStride, // globalRankStride
        this->getSize()); // worldSize

    // avoidRecordStreams_ note: collective() will stash inputTensors and
    // outputTensors.
    return collective(
        inputTensor,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          std::vector<size_t> send_lengths(size_);
          std::vector<size_t> recv_lengths(size_);
          std::vector<size_t> send_offsets(size_);
          std::vector<size_t> recv_offsets(size_);
          c10d::computeLengthsAndOffsets(
              inputSplitSizes, input, &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(
              outputSplitSizes, output, &recv_lengths, &recv_offsets);
          // See [Sync Streams].
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          torch::cuda::nccl::all2all_single_unequal_split(
              input.data_ptr(),
              send_lengths.data(),
              send_offsets.data(),
              output.data_ptr(),
              recv_lengths.data(),
              recv_offsets.data(),
              input.element_size(),
              input.scalar_type(),
              comm,
              stream);
          return ncclSuccess;
        },
        OpType::ALLTOALL_BASE,
        "nccl:all_to_all");
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& /* unused */) {
  std::vector<int64_t> inSplitSizes;
  std::vector<int64_t> outSplitSizes;
  int64_t total_numel = 0;

  auto device = outputTensors[0].device();
  for (const auto r : c10::irange(outputTensors.size())) {
    check_gpu_single_tensor(outputTensors[r], true);
    check_gpu_single_tensor(inputTensors[r], true);
    TORCH_CHECK(
        device == outputTensors[r].device() &&
            device == inputTensors[r].device(),
        "Tensors must be on the same device")
    inSplitSizes.push_back(inputTensors[r].numel());
    outSplitSizes.push_back(outputTensors[r].numel());
    total_numel += inputTensors[r].numel();
  }

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "all_to_all", // collective name
      total_numel, // inNelems
      total_numel, // outNelems
      inputTensors.front().scalar_type(), // dType
      inSplitSizes, // inSplitSizes
      outSplitSizes, // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  return collective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        torch::cuda::nccl::all2all(outputTensors, inputTensors, comm, stream);
        return ncclSuccess;
      },
      [&](at::cuda::CUDAStream&,
          c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
        if (avoidRecordStreams_) {
          // inputTensor0 and outputTensor0 are stashed redundantly by
          // collective(), but that's ok.
          auto& v = work->stashed_for_allocator_safety_;
          v->insert(v->end(), inputTensors.begin(), inputTensors.end());
          v->insert(v->end(), outputTensors.begin(), outputTensors.end());
        }
      },
      [](at::cuda::CUDAStream&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      OpType::ALLTOALL,
      "nccl:all_to_all");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /* unused */) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  check_gpu_single_tensor(tensor, true);

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      dstRank, // dst rank
      "send", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  auto ret = pointToPoint(
      tensor,
      [&](sendImpl(at::Tensor& input,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream,
          int dst),
      dstRank,
      OpType::SEND,
      c10::str("nccl:send ", rank_, "->", dstRank).c_str());
  return ret;
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /* unused */) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  check_gpu_single_tensor(tensor, true);

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      srcRank, // src rank
      "recv", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  auto ret = pointToPoint(
      tensor,
      [&](recvImpl(at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream,
          int src),
      srcRank,
      OpType::RECV,
      c10::str("nccl:recv ", rank_, "<-", srcRank).c_str());
  return ret;
}

void ProcessGroupNCCL::groupStart() {
  groupStartImpl();
  ++ncclActiveGroupCounter_;
}

void ProcessGroupNCCL::groupEnd() {
  groupEndImpl();
  --ncclActiveGroupCounter_;
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupNCCL::gather: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);

  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto inputTensor = inputTensors.back();

  std::vector<at::Tensor> outputs;

  if (getRank() == opts.rootRank) {
    if (outputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element output list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (outputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect output list size " << outputTensors[0].size()
         << ". Output list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = inputTensor.options();
    const auto& sizes = inputTensor.sizes();
    assertTypeAndSizesMatch(invalidArgument, outputTensors[0], options, sizes);
    outputs = outputTensors[0];
  } else {
    // if not in the root rank, initialize outputs as empty list
    if (outputTensors.size() != 0) {
      invalidArgument("requires empty output on non-root");
    }
    outputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    outputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      opts.rootRank, // root rank
      "gather", // collective name
      inputTensor.numel(), // inNelems
      inputTensor.numel() * this->getSize(), // outNelems
      inputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash inputTensors and
  // outputs, which == outputTensors[0] on the root rank where it matters.

  auto inputs = std::vector<at::Tensor>{inputTensor};
  return collective(
      inputs,
      outputs, // just to fit the collective interface
      [&](gather_impl(at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          c10::Stream& stream),
      [](c10::Stream&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
            const auto root = opts.rootRank;
            if (getRank() == root) {
              if (!avoidRecordStreams_) {
                for (auto output : outputs) {
                  c10::cuda::CUDACachingAllocator::recordStream(
                      output.storage().data_ptr(), stream);
                }
              }
            }
         },
      [](c10::Stream&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      OpType::GATHER,
      "nccl:gather");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupNCCL::scatter: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);

  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto outputTensor = outputTensors.back();

  std::vector<at::Tensor> inputs;

  if (getRank() == opts.rootRank) {
    if (inputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element input list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (inputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect input list size " << inputTensors[0].size()
         << ". Input list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = outputTensor.options();
    const auto& sizes = outputTensor.sizes();
    assertTypeAndSizesMatch(invalidArgument, inputTensors[0], options, sizes);
    inputs = inputTensors[0];
  } else {
    // if not in the root rank, initialize inputTensors as empty place holder
    // with an empty list
    if (inputTensors.size() != 0) {
      invalidArgument("requires empty input on non-root");
    }
    inputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    inputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      opts.rootRank, // root rank
      "scatter", // collective name
      outputTensor.numel() * this->getSize(), // inNelems
      outputTensor.numel(), // outNelems
      outputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash outputTensors and
  // inputs, which == inputTensors[0] on the root rank where it matters.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  const auto root = opts.rootRank;
  bool nanCheck = (rank_ == root);

  auto outputs = std::vector<at::Tensor>{outputTensor};
  return collective(
      outputs,
      inputs, // just to fit the collective interface
      [&](scatterImpl(at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          c10::Stream& stream),
      [](c10::Stream&,
         c10::intrusive_ptr<ProcessGroupGCCL::WorkGCCL>& work) {
          if (getRank() == root) {
              if (!avoidRecordStreams) {
                for (auto input : inputs) {
                  c10::cuda::CUDACachingAllocator::recordStream(
                      input.storage().data_ptr(), stream);
                }
              }
          }
         },
      [](c10::Stream&,
         c10::intrusive_ptr<ProcessGroupGCCL::WorkGCCL>& work) {},
      OpType::SCATTER,
      "nccl:scatter",
      avoidRecordStreams,
      nanCheck);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  C10_THROW_ERROR(
      NotImplementedError, "ProcessGroupNCCL does not support recvAnysource");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  check_gpu_single_tensor(input_tensor);
  check_gpu_single_tensor(output_tensor);

  if (input_tensor.dtype() != output_tensor.dtype()) {
    C10_THROW_ERROR(
        TypeError, "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "output tensor size must be equal to world_size times input tensor size");
  }

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      input_tensor, // inputTensors
      output_tensor, // outputTensors
      rank_, // rank
      "_allgather_base", // collective name
      input_tensor.numel(), // inNelems
      output_tensor.numel(), // outNelems
      output_tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash inputs and outputs.
  // Note 2: for asyncOp = false, we don't want to record streams because we
  // know that the NCCL stream will join back to the "current" stream right
  // after this op. So we might just as well keep the stream ownership of the
  // input/output tensors unchanged. The benefit would be that the
  // allocation/free of the tensors would look deterministic to the "current"
  // stream so that the caching allocator can reuse memory pool for this stream
  // in a clever way. This setting is added for libraries like FSDP which uses
  // `all_gather_into_tensor`.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      input_tensor,
      output_tensor,
      [&](allgather_impl(at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          c10::Stream& stream),
      [&]{
          if (!avoidRecordStreams) {
               c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
      },
      OpType::_ALLGATHER_BASE,
      "nccl:_all_gather_base",
      avoidRecordStreams);
}
} // namespace c10d

#endif // USE_C10D_NCCL
