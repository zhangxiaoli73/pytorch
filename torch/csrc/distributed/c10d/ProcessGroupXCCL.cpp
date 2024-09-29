#include <torch/csrc/distributed/c10d/ProcessGroupXCCL.hpp>
#include <fstream>
#include <mutex>
#include <sstream>

#ifdef USE_C10D_XCCL
#include <exception>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <ATen/detail/FunctionTraits.h>
#include <c10/core/DeviceType.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/torch.h>

namespace c10d {

namespace {
std::map<c10d::ReduceOp, ccl::reduction> xcclOps = {
    {ReduceOp::MIN, ccl::reduction::min},
    {ReduceOp::MAX, ccl::reduction::max},
    {ReduceOp::SUM, ccl::reduction::sum},
    {ReduceOp::PRODUCT, ccl::reduction::prod},
};

std::map<at::ScalarType, ccl::datatype> xcclDatatypes = {
    {at::kByte, ccl::datatype::uint8},
    {at::kChar, ccl::datatype::int8},
    {at::kInt, ccl::datatype::int32},
    {at::kLong, ccl::datatype::int64},
    {at::kHalf, ccl::datatype::float16},
    {at::kFloat, ccl::datatype::float32},
    {at::kDouble, ccl::datatype::float64},
    {at::kBFloat16, ccl::datatype::bfloat16},
    {at::kBool, ccl::datatype::uint8},
};

XCCL_KVS kvs;
std::mutex kvs_mutex;

XCCL_KVS get_kvs(int rank, c10d::Store& store) {
  std::lock_guard<std::mutex> lock(kvs_mutex);
  if (kvs)
    return kvs;
  std::string storeKey = "xccl_kvs";

  // Rank 0 broadcast the bootstrap network information to other ranks
  if (rank == 0) {
    kvs = ccl::create_main_kvs();
    ccl::kvs::address_type main_addr = kvs->get_address();
    auto ccl_kvs_addr =
        std::vector<uint8_t>(main_addr.begin(), main_addr.end());
    store.set(storeKey, ccl_kvs_addr);
  } else {
    auto ccl_kvs_addr = store.get(storeKey);
    if (ccl_kvs_addr.size() != ccl::kvs::address_max_size) {
      throw std::runtime_error("Unexpected ccl kvs addr from the store\n");
    }
    ccl::kvs::address_type main_addr;
    std::copy_n(
        ccl_kvs_addr.begin(), ccl::kvs::address_max_size, main_addr.begin());
    kvs = ccl::create_kvs(main_addr);
  }

  return kvs;
}

void check_xpu_single_tensor(const at::Tensor& tensor) {
  if (!tensor.is_xpu() || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be XPU and dense");
  }
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
  }
}

ccl::datatype getXcclDataType(at::ScalarType type) {
  auto it = xcclDatatypes.find(type);
  TORCH_CHECK_WITH(
      TypeError,
      it != xcclDatatypes.end(),
      "Input tensor data type is not supported for XCCL process group: ",
      type);
  return it->second;
}

ccl::reduction getXcclReduceOp(const ReduceOp& reduceOp, at::Tensor& input) {
  try {
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == ReduceOp::SUM) {
        // For bool tensors, map sum to max, which both represent a bitwise or.
        // This is to prevent overflow issues with sum, since we use uint8 to
        // represent a bool (see xcclDatatypes mapping align with cuda).
        return ccl::reduction::max;
      }
    }
    return xcclOps.at(reduceOp);
  } catch (const std::out_of_range&) {
    switch (reduceOp) {
      case ReduceOp::AVG:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp AVG with XCCL");
        break;
      case ReduceOp::BAND:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BAND with XCCL");
        break;
      case ReduceOp::BOR:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BOR with XCCL");
        break;
      case ReduceOp::BXOR:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BXOR with XCCL");
        break;
      default:
        C10_THROW_ERROR(ValueError, "Unhandled ReduceOp");
        break;
    }
  }
}
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
  ncclEvent.record(at::xpu::getCurrentXPUStream(device.index()));
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
static std::unordered_map<std::shared_ptr<void*>, int> ncclCommDevIdxMap;
static std::mutex ncclCommDevIdxMapMutex;
static bool allocatorHooksAttached = false;

constexpr int64_t kSynchronizeBusyWaitMillis = 10;
thread_local uint64_t ProcessGroupXCCL::ncclActiveGroupCounter_ = 0;

std::ostream& operator<<(
    std::ostream& output,
    const ProcessGroupXCCL::WorkXCCL& WorkXCCL) {
  std::string workInfo;
  return output << workInfo;
}

ProcessGroupXCCL::WorkXCCL::WorkXCCL(
    const std::string& pgUID,
    const std::string& pgDesc,
    at::Device& device,
    int rank,
    OpType opType,
    uint64_t seq,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputs,
    bool enableTiming,
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
//  if (cudaEventCacheEnabled) {
//    ncclStartEvent_ = enableTiming
//        ? ProcessGroupXCCL::CUDAEventCache::get().create(enableTiming)
//        : nullptr;
//    ncclEndEvent_ =
//        ProcessGroupXCCL::CUDAEventCache::get().create(enableTiming);
//  } else {
//    ncclStartEvent_ = enableTiming
//        ? std::make_shared<at::cuda::CUDAEvent>(cudaEventDefault)
//        : nullptr;
//    ncclEndEvent_ = std::make_shared<at::cuda::CUDAEvent>(
//        enableTiming ? cudaEventDefault : cudaEventDisableTiming);
//  }
  unsigned char enable_timing = 0;
  ncclStartEvent_ = std::make_shared<at::xpu::XPUEvent>(enable_timing);
  ncclEndEvent_ = std::make_shared<at::xpu::XPUEvent>(enable_timing);
}

ProcessGroupXCCL::WorkXCCL::WorkXCCL(const WorkXCCL& w)
    : Work(w.rank_, w.opType_),
      std::enable_shared_from_this<WorkXCCL>(w),
      pgUID_(w.pgUID_),
      pgDesc_(w.pgDesc_),
      device_(w.device_),
      ncclStartEvent_(w.ncclStartEvent_),
      ncclEndEvent_(w.ncclEndEvent_),
      cclComm_(w.cclComm_),
      blockingWait_(w.blockingWait_),
      opTimeout_(w.opTimeout_),
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

ProcessGroupXCCL::WorkXCCL::~WorkXCCL() = default;

bool ProcessGroupXCCL::WorkXCCL::isCompleted() {
    return true;
}

bool ProcessGroupXCCL::WorkXCCL::isSuccess() const {
  C10_THROW_ERROR(NotImplementedError, "Work::isSuccess() is deprecated");
}

const std::string& ProcessGroupXCCL::WorkXCCL::logPrefix() const {
  static std::string prefix = c10::str("[Rank ", rank_, "] ");
  return prefix;
}

bool ProcessGroupXCCL::WorkXCCL::checkTimeout(
    std::optional<std::chrono::milliseconds> timeout) {
  STATIC_SCOPED_WAIT_COUNTER(
      pytorch.wait_counter.ProcessGroupXCCL__checkTimeout);
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

void ProcessGroupXCCL::WorkXCCL::handleException(
    ErrorHandlingMode errorHandling) {
  if (exception_) {
    auto exceptionMsg = c10::str(
        "Some NCCL operations have failed or timed out. Due to the ",
        "asynchronous nature of CUDA kernels, subsequent GPU operations ",
        "might run on corrupted/incomplete data.");
    LOG(ERROR) << logPrefix() << exceptionMsg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupXCCL.WorkXCCL.handleException");

    if (SHOULD_TEAR_DOWN(errorHandling)) {
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process down.");
      LOG(ERROR) << logPrefix() << tearDownMsg;
      std::rethrow_exception(exception_);
    }
  }
}

void ProcessGroupXCCL::WorkXCCL::synchronize() {
  // Call Synchronize without a timeout. We use this method to avoid adding a
  // timeout argument to the public synchronize API.
  synchronizeInternal(kNoTimeout);
}

void ProcessGroupXCCL::WorkXCCL::synchronizeStream() {
  auto currentStream = at::xpu::getCurrentXPUStream(device_.index());
  // Block the current stream on the NCCL stream
  ncclEndEvent_->block(currentStream);

  if (avoidRecordStreams_) {
    stashed_for_allocator_safety_->clear();
  }
}

// Waiting on the work's corresponding CUDA events
void ProcessGroupXCCL::WorkXCCL::synchronizeInternal(
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
     this->abort();
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
    auto currentStream = at::xpu::getCurrentXPUStream(device_.index());
    // CUDAStream wrapper will correctly use a DeviceGuard here
    currentStream.synchronize();
  }
}

// Same as calling synchronize().
bool ProcessGroupXCCL::WorkXCCL::wait(std::chrono::milliseconds timeout) {
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

void ProcessGroupXCCL::WorkXCCL::abort() {
  // Abort all communicators of this work
//  ncclCommAbort(cclComm_);

  ncclCommDevIdxMapMutex.lock();
  ncclCommDevIdxMap.erase(cclComm_);
  ncclCommDevIdxMapMutex.unlock();
}

static std::atomic<size_t> process_group_id = 0;

constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple. You are probably using multiple "
    "devices under one thread. The support for such usage has been deprecated. "
    "For details, please refer to "
    "https://pytorch.org/docs/stable/distributed.html#multi-gpu-collective-functions. "
    "ProcessGroupXCCL continues supporting multi-process and multi-thread modes.";

ProcessGroupXCCL::ProcessGroupXCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(options),
      ncclCommCounter_(0),
      terminateProcessGroup_(false),
      collectiveDebugInfoMode_(false),
      local_id_(process_group_id++) {
  init();
   // Intel oneCCL requires passing CCL_LOCAL_RANK and CCL_LOCAL_SIZE for non-MPI
  // launchers.
  if (!with_mpirun()) {
    int local_rank = getXCCLEnvVar("LOCAL_RANK");
    int local_world_size = getXCCLEnvVar("LOCAL_WORLD_SIZE");
    if (local_rank == -1 || local_world_size == -1) {
      local_rank = rank;
      local_world_size = size;
    }
    setXCCLEnvVar("CCL_PROCESS_LAUNCHER", "none");
    setXCCLEnvVar("CCL_LOCAL_RANK", local_rank);
    setXCCLEnvVar("CCL_LOCAL_SIZE", local_world_size);
  }
}

void ProcessGroupXCCL::setSequenceNumberForGroup() {
} // NCCL just starts sequence numbers at 0.

uint64_t ProcessGroupXCCL::getSequenceNumberForGroup() {
  return seqCollective_;
}

void ProcessGroupXCCL::enableCollectivesTiming() {
  enableTiming_.store(true);
}

void ProcessGroupXCCL::waitForFutureOrTimeout(
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
    data.strings["flight_recorder_version"] = ""; //c10d::version_val_str;
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
// Abort all communicators on this rank
bool ProcessGroupXCCL::abort(std::optional<std::string> abortReason) {
  return true;
}
// todo: do we need it?
void ProcessGroupXCCL::shutdown(std::optional<std::string> reason) {
  // Don't join threads here since the purpose of this method is to abort all
  // communicators and signal the threads to exit. Joining on the threads could
  // potentially block and hence avoid it in this method.
  terminateProcessGroup_.store(true);
  workMetaListCV_.notify_one();

  // lauch abort asynchrounously and wait for it to complete or timeout
  LOG(INFO) << logPrefix()
            << "Launching ProcessGroupXCCL abort asynchrounously.";
  std::future<bool> fut = std::async(
      std::launch::async, [this, &reason]() { return this->abort(reason); });

  waitForFutureOrTimeout(fut, options_->timeout, "ProcessGroup abort", true, false);
  LOG(INFO) << logPrefix() << "ProcessGroupXCCL aborts successfully.";

  // We need to wait for abort to finish before we can safely shut down
  // heartbeat monitoring thread.
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();
}

ProcessGroupXCCL::~ProcessGroupXCCL() {
  LOG(INFO) << logPrefix() << "ProcessGroupXCCL destructor entered.";

  if (!terminateProcessGroup_.load()) {
    if (rank_ % localDeviceCount_ == 0) {
      TORCH_WARN_ONCE(
          "WARNING: process group has NOT been destroyed before we destruct ProcessGroupXCCL. ",
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

// We want to have both PG ID and global unique ID (guid) for the logging
// prefix. PG ID records how many ProcessGroupXCCL objects were created on a
// specific rank and is a stable index across ranks, which lets users reason
// about, for example, the second PG we initialized on this rank is for FSDP,
// and corresponds with PG ID = 1 on other ranks as well. Unlike PG ID, guid (or
// group name) is a global unique ID across ranks. The guid is either a hash of
// all the ranks in the group or a counter of how many times
// `_process_group_name` is called, essentially it means how many times we
// have PGs users have created. Before using split_group, even if
// we are creating a new sub-PG, all ranks have to call the API at the same
// time, and this makes `group_name` a unique identifier for a group (PG).
std::string ProcessGroupXCCL::createLogPrefix() const {
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

const std::string& ProcessGroupXCCL::logPrefix() const {
    return "";
//  return logPrefix_;
}

const int& ProcessGroupXCCL::globalRank() const {
  static int globalRank = rank_;
  return globalRank;
}

const std::vector<uint64_t>& ProcessGroupXCCL::groupRanks() const {
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
    if (!t.is_xpu() || t.is_sparse()) {
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

c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> ProcessGroupXCCL::initWork(
    at::Device& device,
    int rank,
    OpType opType,
    const char* profilingTitle,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs, // TODO(kwen2501): necessary?
    bool record) {
  auto r = c10::make_intrusive<ProcessGroupXCCL::WorkXCCL>(
      pg_uid_,
      pg_desc_,
      device,
      rank,
      opType,
      seqCollective_,
      profilingTitle,
      profilingTitle != nullptr ? std::optional<std::vector<at::Tensor>>(inputs)
                                : std::nullopt,
      enableTiming_.load(),
      dist_debug_level_);
  return r;
}

// TODO(kwen2501): deprecate
std::vector<at::Tensor> ProcessGroupXCCL::WorkXCCL::result() {
  return *outputs_;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupXCCL::WorkXCCL::
    getFuture() {
  return future_;
}

float ProcessGroupXCCL::WorkXCCL::getDuration() const {
  TORCH_CHECK(timingEnabled_, "getDuration only works if timing was enabled");
  TORCH_CHECK(
      ncclStartEvent_,
      "getDuration only works if ncclStartEvents_ is populated, true if timing enabled");
  TORCH_CHECK(
      ncclEndEvent_,
      "getDuration only works if ncclEndEvents_ is populated, which should always be true");
//  return ncclStartEvent_->elapsed_time(*ncclEndEvent_);
  return 0.0f;
}

uint64_t ProcessGroupXCCL::WorkXCCL::getSequencenumber() const {
  return seq_;
}

void ProcessGroupXCCL::assignTimeoutToWork(
    const c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work,
    const c10::intrusive_ptr<ProcessGroupXCCL::Options>& option) {
  std::chrono::milliseconds timeout = option->timeout;
  work->opTimeout_ = timeout;
}

void ProcessGroupXCCL::workEnqueue(
    c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work) {
  if (!terminateProcessGroup_.load()) {
    std::lock_guard<std::mutex> lock(workMetaListMutex_);
    // Avoid view tensors to be processed in cleanup thread.
    // View tensors' destruction invokes autograd_meta, which
    // needs to be destructed in user thread. Otherwise will
    // get deadlock. Here we enqueue work without outputs_.
    workMetaList_.emplace_back(*work);
//    // update the PG status related to the last enqueued work
//    pgStatus_->lastEnqueuedSeq = work->seq_;
//    pgStatus_->lastEnqueuedWorkName = opTypeToString(work->opType_);
//    pgStatus_->lastEnqueuedNumelIn = work->numelIn_;
//    pgStatus_->lastEnqueuedNumelOut = work->numelOut_;
    lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
  }
}

ProcessGroupXCCL::Options::Options(bool is_high_priority_stream)
    : Backend::Options(XCCL_BACKEND_NAME, kProcessGroupXCCLDefaultTimeout),
      is_high_priority_stream(is_high_priority_stream) {}

static constexpr int CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;

void ProcessGroupXCCL::startCoalescing() {
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
c10::intrusive_ptr<Work> ProcessGroupXCCL::endCoalescing(OpType optype) {
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

//  // Create Work object
//  c10::cuda::CaptureStatus capture_status =
//      c10::cuda::currentStreamCaptureStatusMayInitCtx();
  bool enqueue = false;
//      (coalescing_state_) && capture_status == c10::cuda::CaptureStatus::None;
  auto work =
      initWork(device, rank_, optype, "nccl:coalesced", {}, {}, enqueue);

  work->cclComm_ = comm;
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
//  at::cuda::CUDAGraph::inc_pending_event_queries();
//
//  if (enqueue) {
//    workEnqueue(work);
//  } else {
//    at::cuda::CUDAGraph::dec_pending_event_queries();
//  }

   workEnqueue(work);

  coalescing_state_ = 0;
  coalescedComm_ = nullptr;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::endCoalescing() {
  // Default OpType to COALESCED if not specified
  return endCoalescing(OpType::COALESCED);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce_sparse(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
    TORCH_CHECK(
        false, "ProcessGroupXCCL::allreduce_sparse not implemented");
}

void ProcessGroupXCCL::CollectivesImpl::allreduceImpl(at::Tensor& input, at::Tensor& output,
    cclComm_t comm, c10::Stream& stream, const AllreduceOptions& opts) {
    // comm created
    auto xcclDataType = getXcclDataType(input.scalar_type());
    auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
    ccl::event ret_evt;
//    ret_evt = ccl::allreduce(
//        input.data_ptr(),
//        output.data_ptr(),
//        (size_t)input.numel(),
//        xcclDataType,
//        xcclReduceOp,
//        comm,
//        stream);
    return;
}

bool ProcessGroupXCCL::complexViewAsRealAllowed(const ReduceOp reduceOp) {
  return false;
}

//  template <typename Fn, typename PreProcess, typename PostProcess>
//   c10::intrusive_ptr<Work> ProcessGroupXCCL::collective(
//      std::vector<at::Tensor>& inputs,
//      std::vector<at::Tensor>& outputs,
//      Fn fn,
//      PreProcess pre,
//      PostProcess post,
//      OpType opType,
//      const char* profilingTitle = nullptr,
//      bool avoidRecordStreams = false,
//      bool nanCheck = true) {
//      using traits = function_traits<Fn>;
//      using attr_t = typename traits::template arg<2>::type;
//      attr_t attr = ccl::create_operation_attr<attr_t>();
//
//      auto device = inputs[0].device();
//      const auto key = std::to_string(device.index());
//      auto comm = getXCCLComm(key, device);
//
//      auto stream = xcclStreams_.at(key);
//
//      c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;
//
//      work = initWork(device, rank_, opType, profilingTitle, avoidRecordStreams, nanCheck);
//
//      work->outputs_ =
//          std::make_shared<std::vector<at::Tensor>>(std::move(outputs));
//      c10::xpu::XPUCachingAllocator::recordStream(
//          input.storage().data_ptr(), stream);
//
//      auto ccl_stream = ccl::create_stream(stream.queue());
//
//      fn(input, output, attr, *comm, ccl_stream);
//
//      work->ncclEndEvent_->record(stream);
//
//      std::vector<c10::Stream> streams = {stream.unwrap()};
//      c10::MultiStreamGuard streamGuard(streams);
//      std::vector<at::Device> devices{device};
//      work->future_ = c10::make_intrusive<at::ivalue::Future>(
//          c10::ListType::create(c10::TensorType::get()), devices);
//      work->future_->markCompleted(at::IValue(*work->outputs_));
//      work->blockingWait_ = blockingWait_;
//
//      return work;
//   }

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce(
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

  // @lint-ignore CLANGTIDY
//  RECORD_PARAM_COMMS_DATA(
//      static_cast<int>(
//          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
//      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
//      tensors, // inputTensors
//      tensors, // outputTensors
//      rank_, // rank
//      "allreduce", // collective name
//      tensor.numel(), // inNelems
//      tensor.numel(), // outNelems
//      tensor.scalar_type(), // dType
//      std::vector<int64_t>(), // inSplitSizes
//      std::vector<int64_t>(), // outSplitSizes
//      globalRankStart, // globalRankStart
//      globalRankStride, // globalRankStride
//      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return collective(
      tensor,
      tensor,
      CollectivesImpl::allreduceImpl,
      OpType::ALLREDUCE,
      "xccl:all_reduce");
  }
} // namespace c10d

#endif // USE_C10D_NCCL
