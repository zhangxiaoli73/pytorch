#ifdef USE_C10D_XCCL

#include <exception>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <c10/core/DeviceType.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/WaitCounter.h>
#include <c10/util/irange.h>
#include <c10/util/thread_name.h>
#include <torch/csrc/distributed/c10d/NanCheck.hpp>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupCCL.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/torch.h>
#include <optional>

namespace c10d {

namespace {
    // Returns exception's what() given an exception_ptr instance.
    std::string getExceptionMsgFromExceptionPtr(
        const std::exception_ptr& exceptionPtr) {
      TORCH_CHECK(exceptionPtr != nullptr);
      try {
        std::rethrow_exception(exceptionPtr);
      } catch (const std::exception& e) {
        return e.what();
      } catch (...) {
        return "Unknown exception type";
      }
    }
}
constexpr int64_t kSynchronizeBusyWaitMillis = 10;

std::ostream& operator<<(
    std::ostream& output,
    const ProcessGroupCCL::WorkCCL& WorkCCL) {
  std::string workInfo;
  workInfo = c10::str(
      "WorkCCL(",
      "SeqNum=",
      WorkCCL.seq_,
      ", OpType=",
      opTypeToString(WorkCCL.opType_),
      ", NumelIn=",
      WorkCCL.numelIn_,
      ", NumelOut=",
      WorkCCL.numelOut_,
      ", Timeout(ms)=",
      WorkCCL.opTimeout_.count(),
      ")");
  return output << workInfo;
}

ProcessGroupCCL::WorkCCL::WorkCCL(
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
//  if (cudaEventCacheEnabled) {
//    cclStartEvent_ = enableTiming
//        ? ProcessGroupCCL::AccEventCache::get().create(enableTiming)
//        : nullptr;
//    cclEndEvent_ =
//        ProcessGroupCCL::AccEventCache::get().create(enableTiming);
//  } else {
//  // todo: zl_debug what's the cudaEventDefault?
//    cclStartEvent_ = enableTiming
//        ? std::make_shared<c10::Event>(cudaEventDefault)
//        : nullptr;
//    cclEndEvent_ = std::make_shared<c10::Event>(
//        enableTiming ? cudaEventDefault : cudaEventDisableTiming);
//  }
  cclStartEvent_ =  nullptr;
  cclEndEvent_ = std::shared_ptr<c10::Event>(new c10::Event(device.type()));
}

ProcessGroupCCL::WorkCCL::WorkCCL(const WorkCCL& w)
    : Work(w.rank_, w.opType_),
      std::enable_shared_from_this<WorkCCL>(w),
      pgUID_(w.pgUID_),
      pgDesc_(w.pgDesc_),
      device_(w.device_),
      cclStartEvent_(w.cclStartEvent_),
      cclEndEvent_(w.cclEndEvent_),
      cclComm_(w.cclComm_),
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

ProcessGroupCCL::WorkCCL::~WorkCCL() = default;

bool ProcessGroupCCL::WorkCCL::isCompleted() {
  if (!cclCommIsAborted) {
    checkAndSetException();
  }
  return exception() || finishedGPUExecutionInternal();
}

bool ProcessGroupCCL::WorkCCL::isStarted() {
  if (!cclCommIsAborted) {
    checkAndSetException();
  }
  return exception() || startedGPUExecutionInternal();
}

bool ProcessGroupCCL::WorkCCL::isSuccess() const {
  C10_THROW_ERROR(NotImplementedError, "WorkCCL::isSuccess() is deprecated");
}

void ProcessGroupCCL::WorkCCL::checkAndSetException() {
  if (exception()) {
    // We already have an exception.
    return;
  }

  auto exception_ptr = checkForCCLErrors();
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
  if (exception_) {
    LOG(ERROR) << logPrefix() << "Collective " << *this
               << " raised the following async exception: "
               << getExceptionMsgFromExceptionPtr(exception_);
  }
}

const std::string& ProcessGroupCCL::WorkCCL::logPrefix() const {
  static std::string prefix = c10::str("[Rank ", rank_, "] ");
  return prefix;
}

void ProcessGroupCCL::WorkCCL::setException(
    std::exception_ptr exception_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
}

// Helper that checks if the ccl kernels are completed on the GPUs
bool ProcessGroupCCL::WorkCCL::finishedGPUExecution() {
  checkAndSetException();
  return finishedGPUExecutionInternal();
}

bool ProcessGroupCCL::WorkCCL::startedGPUExecutionInternal() const {
  // if timing is disabled we won't have allocated start events
  if (!timingEnabled_) {
    return false;
  }
  // Checking the work's corresponding CUDA event's status
  if (!cclStartEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupCCL::WorkCCL::finishedGPUExecutionInternal() const {
  // Checking the work's corresponding CUDA event's status
  // It calls `cudaEventQuery` eventually. Although this seems to be a
  // non-blocking call, but we did notice hangs in the past. It can
  // hang if another thread is holding the CUDA global context lock. For
  // example, when doing a `cudaDeviceSynchronize` or even
  // `cudaStreamSynchronize`.
  if (!cclEndEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupCCL::WorkCCL::checkTimeout(
    std::optional<std::chrono::milliseconds> timeout) {
  STATIC_SCOPED_WAIT_COUNTER(
      pytorch.wait_counter.ProcessGroupCCL__checkTimeout);
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

void ProcessGroupCCL::WorkCCL::handleException(
    ErrorHandlingMode errorHandling) {
  if (exception_) {
    auto exceptionMsg = c10::str(
        "Some ccl operations have failed or timed out. Due to the ",
        "asynchronous nature of CUDA kernels, subsequent GPU operations ",
        "might run on corrupted/incomplete data.");
    LOG(ERROR) << logPrefix() << exceptionMsg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupCCL.WorkCCL.handleException");

    if (SHOULD_TEAR_DOWN(errorHandling)) {
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process down.");
      LOG(ERROR) << logPrefix() << tearDownMsg;
      std::rethrow_exception(exception_);
    }
  }
}

void ProcessGroupCCL::WorkCCL::synchronize() {
  // Call Synchronize without a timeout. We use this method to avoid adding a
  // timeout argument to the public synchronize API.
  synchronizeInternal(kNoTimeout);
}

void ProcessGroupCCL::WorkCCL::synchronizeStream() {
  // todo: zl_debug change to more generic
  // auto currentStream = at::xpu::getCurrentXPUStream(device_.index());
  c10::impl::VirtualGuardImpl impl(device_.type());
  c10::Stream currentStream = impl.getStream(device_);

  // Block the current stream on the ccl stream
  cclEndEvent_->block(currentStream);

  if (avoidRecordStreams_) {
    stashed_for_allocator_safety_->clear();
  }
}

// Waiting on the work's corresponding CUDA events
void ProcessGroupCCL::WorkCCL::synchronizeInternal(
    std::chrono::milliseconds timeout) {
  synchronizeStream();

  // In case of blocking, wait for the operation to complete.
  if (blockingWait_) {
    while (!isCompleted()) {
      bool timedOut = checkTimeout(
          timeout == kNoTimeout ? std::nullopt : std::make_optional(timeout));
      // Explicitly abort cclComms here before throwing this timed out
      // exception to users.
      // If throwing timed out excepiton without aborting ccl communicators
      // here, it was observed that CUDA GPU will have 100% utilization and
      // can not run new events successfully.
      if (timedOut) {
        std::string exceptionMsg = c10::str(
            logPrefix(),
            "Work ",
            (*this),
            " timed out in blocking wait (TORCH_ccl_BLOCKING_WAIT=1).");
        LOG(ERROR) << exceptionMsg;
        break;
      }
      // Yield
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
    // exception() includes timeout and error during blocking wait
    if (exception()) {
      // Abort ccl communicators
      abort();
      // Throw exception (from main thread here)
      handleException(TearDown);
    }
  }
}

// Same as calling synchronize().
bool ProcessGroupCCL::WorkCCL::wait(std::chrono::milliseconds timeout) {
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
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupCCL::WorkCCL::abort() {
  // Abort all communicators of this work
  cclCommAbort(cclComm_);
}


static std::atomic<size_t> process_group_id = 0;

constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple. You are probably using multiple "
    "devices under one thread. The support for such usage has been deprecated. "
    "For details, please refer to "
    "https://pytorch.org/docs/stable/distributed.html#multi-gpu-collective-functions. "
    "ProcessGroupCCL continues supporting multi-process and multi-thread modes.";

std::exception_ptr ProcessGroupCCL::WorkCCL::checkForCCLErrors() {
  return checkForCCLErrorsInternal(cclComm_);
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupCCL::WorkCCL::
    getFuture() {
  return future_;
}

float ProcessGroupCCL::WorkCCL::getDuration() const {
  TORCH_CHECK(timingEnabled_, "getDuration only works if timing was enabled");
  TORCH_CHECK(
      cclStartEvent_,
      "getDuration only works if cclStartEvents_ is populated, true if timing enabled");
  TORCH_CHECK(
      cclEndEvent_,
      "getDuration only works if cclEndEvents_ is populated, which should always be true");
  // todo: zl_debug how to use it?
  //return cclStartEvent_->elapsed_time(*cclEndEvent_);
  return 0.0f;
}

uint64_t ProcessGroupCCL::WorkCCL::getSequencenumber() const {
  return seq_;
}

void ProcessGroupCCL::cclCommAbort(std::shared_ptr<cclComm_t> cclComm_) {
//    cclCommIsAborted = true;
    return;
}
std::exception_ptr ProcessGroupCCL::checkForCCLErrorsInternal(std::shared_ptr<cclComm_t> cclComm_) {
    return nullptr;
}

ProcessGroupCCL::ProcessGroupCCL(
    int rank,
    int size)
    : Backend(rank, size){}

ProcessGroupCCL::~ProcessGroupCCL() {
//  LOG(INFO) << logPrefix() << "ProcessGroupCCL destructor entered.";
  LOG(INFO) << "ProcessGroupCCL destructor entered.";
}

} // namespace c10d

#endif // USE_C10D_XCCL
