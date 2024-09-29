#pragma once

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#ifdef USE_C10D_XCCL
#include <ATen/xpu/XPUEvent.h>
#include <oneapi/ccl.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <exception>
#include <memory>
#include <vector>

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <c10/core/StreamGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
//#include <ATen/DynamicLibrary.h>
//#include <c10/core/Stream.h>
//#include <c10/core/StreamGuard.h>
//
//#include <torch/custom_class.h>

typedef void* cclComm_t;

namespace c10d {

namespace {
int getXCCLEnvVar(std::string envVarName) {
  char* stringValue = std::getenv(envVarName.c_str());
  if (stringValue != nullptr) {
    try {
      int val = std::stoi(stringValue);
      return val;
    } catch (std::exception& e) {
      TORCH_CHECK(
          false,
          "Invalid value for environment variable: " + std::string(envVarName));
    }
  } else {
    return -1;
  }
}

template <typename T>
void setXCCLEnvVar(const std::string& envVarName, T val) {
  if constexpr (std::is_same_v<T, int>) {
    setenv(envVarName.c_str(), std::to_string(val).c_str(), 1);
  } else if constexpr (std::is_same_v<T, std::string>) {
    setenv(envVarName.c_str(), val.c_str(), 1);
  }
}

bool with_mpirun() {
  return (getenv("MPI_LOCALRANKID") || getenv("MPI_LOCALNRANKS") ||
          getenv("PMI_RANK") || getenv("PMI_SIZE") || getenv("PMIX_RANK"))
      ? true
      : false;
}
} // namespace

#define SHOULD_TEAR_DOWN(a) (a != NoHandling && a != CleanUpOnly)

using xcclComm_t = ccl::communicator;
using XCCL_KVS = ccl::shared_ptr_class<ccl::kvs>;
constexpr const char* XCCL_BACKEND_NAME = "xccl";

constexpr const int kWorkStatusUpdatePeriodMs = 30 * 1000; // 30 seconds

constexpr auto kProcessGroupXCCLDefaultTimeout =
    std::chrono::milliseconds(10 * 60 * 1000);

enum ErrorHandlingMode {
  NoHandling = 0,
  TearDown = 1,
  CleanUpOnly = 2,
  SkipCleanUp = 3
};

// ProcessGroupXCCL implements NCCL bindings for c10d.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// All NCCL functions provided by this class are asynchronous functions. More
// specifically, each NCCL call is scheduled on a separate CUDA stream that is
// different from the current CUDA stream. This is for the purpose of
// achieving potentially concurrency and better performance. As a result,
// it is the callers' responsibility to make sure that the CUDA stream their
// code works on needs to wait for the NCCL operation from
// this class.
//
// This can be done by calling:
//
// either WorkXCCL::wait() or WorkXCCL::synchronize(), both achieves the same
// functionality and are synonyms.
//
// Also note that WorkXCCL::finishedGPUExecution() is a helper function only
// provided by ProcessGroupXCCL to check if the NCCL operation of WorkXCCL has
// finished execution on the GPU (not just scheduled).
//
// Example on using the NCCL process group
//
//   ProcessGroupXCCL pg(store, rank, size);
//   std::shared_ptr<WorkXCCL> work = pg.allreduce(tensors);
//
//   // At this point, NCCL kernel has already by queued successfully
//   // Now, let current stream wait for the NCCL to finish, this function is
//   // async operation as well
//
//   work->wait()
//
//   // Now continue on other work in the current stream.
class TORCH_API ProcessGroupXCCL : public Backend {
 public:
  class WorkXCCL : public Work {
   public:
    friend struct WorkInfo;

    // Constructor takes a list of CUDA devices
    WorkXCCL(
        const std::string& pgUID,
        const std::string& pgDesc,
        at::Device& device,
        int rank,
        OpType opType,
        uint64_t seq,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputs = std::nullopt,
        bool enableTiming = false,
        DebugLevel distDebugLevel = DebugLevel::Off);

    // Copy constructor doing partial copy without outputs_. Cleanup thread
    // monitors and removes finished works. However it will deadlock when
    // destructs outputs_ tensors who are view tensors in autograd graph.
    WorkXCCL(const WorkXCCL& w);

    ~WorkXCCL() override;

    bool isAborted();

    // Checks if request has completed. In this specific case of NCCL, it checks
    // if the NCCL operation has completed on the GPU in its own NCCL stream.
    // Non-blocking operation.
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for NCCL work.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void abort() override;

    // Let current stream wait on the completing of the NCCL work
    // Throws on exceptions. Blocking operation, which will wait for work
    // completion.
    void synchronize() override;

    // Synchronize streams by blocking each on the NCCL stream
    void synchronizeStream();

    // Helper function to handle exception (throw if needed).
    void handleException(ErrorHandlingMode asyncErrorHandling);

    // Helper function that checks if the NCCL kernels have finished
    // execution on the GPUs
    bool finishedGPUExecution();

    // Get a Future object that will be marked as completed internally.
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    float getDuration() const override;

    uint64_t getSequencenumber() const override;

    const std::string& logPrefix() const;

    // Helper function that sets an exception_ptr on the WorkXCCL object.
    void setException(std::exception_ptr exception_ptr);

    // Helper function that returns True if the WorkXCCL object has timed out
    // and False otherwise.
    // In case of timeout, set exception on the WorkXCCL object.
    bool checkTimeout(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt);

    std::vector<at::Tensor> result() override;

   protected:
    // The process group unique id
    std::string pgUID_;

    // The process group description
    std::string pgDesc_;

    // The cached list of CUDA devices to operate on
    at::Device device_;

    // The start CUDA event of NCCL operator tracking this work item. These
    // start CUDA events are needed by desync debugging if enabled.
    std::shared_ptr<at::xpu::XPUEvent> ncclStartEvent_;

    // The end CUDA event of NCCL operator tracking this work item.
    std::shared_ptr<at::xpu::XPUEvent> ncclEndEvent_;

    // The NCCL communicator used for this work item.
    std::shared_ptr<void*> cclComm_;

    // Tensors used for barrier op
    at::Tensor barrierTensor_;

    // Clone of blockingWait_ from ProcessGroupXCCL.
    bool blockingWait_ = false;

    // Clone of avoidRecordStreams_ from ProcessGroupXCCL.
    bool avoidRecordStreams_ = false;

    // Clone of opTimeout_ from ProcessGroupXCCL.
    std::chrono::milliseconds opTimeout_;

    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    // Record the collective sequential number.
    uint64_t seq_;

    // Indicates if the nccl start event has been updated to the store trace.
    // This will be used by desync debug.
    bool startTraceUpdated_{false};

    // Record collective sizes for debug. We only record the size on the first
    // device as multi-device per process is deprecated
    size_t numelIn_ = -1;
    size_t numelOut_ = -1;

    // Wrapper method for the static checkForNCCLErrors which can be overridden
    // for tests.
    std::exception_ptr checkForNCCLErrors();

    friend std::ostream& operator<<(
        std::ostream& output,
        const WorkXCCL& WorkXCCL);

   private:
    // Helper function for synchronize
    void synchronizeInternal(std::chrono::milliseconds timeout);

    // Checks for NCCL errors and sets an appropriate exception_ptr.
    void checkAndSetException();

    // Just checks whether GPU execution has started, without modifying
    // exception_ptr.
    bool startedGPUExecutionInternal() const;

    // Just checks whether GPU execution has completed, without modifying
    // exception_ptr.
    bool finishedGPUExecutionInternal() const;

    // Reference to the store so that we can write aborted communicators
    // to the store.
    c10::intrusive_ptr<Store> store_;

    // Store a reference to NCCL collective's outputs, used by result and to
    // give a more descriptive message when representing the Work as a string.
    std::shared_ptr<std::vector<at::Tensor>> outputs_;

    // TORCH_NCCL_AVOID_RECORD_STREAMS implementation helper.
    // Stores references to participating non-output tensors (ie inputs,
    // flattened intermediates).
    // We'll clear this list in synchronizeStream, just after user-facing
    // stream(s) are synced with the nccl work stream(s).
    // By keeping these refs (as well as outputs_) alive until after the
    // collective's work rejoins the user-facing streams, we achieve
    // caching allocator safety without any recordStream calls.
    // For in-place collectives, some refs stashed here may alias outputs_,
    // but that doesn't do any harm.
    std::shared_ptr<std::vector<at::Tensor>> stashed_for_allocator_safety_;

    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;

    bool timingEnabled_;
    // unique id used to tell the trace buffer that this
    // work has completed
    std::optional<uint64_t> trace_id_;
    DebugLevel distDebugLevel_;
    friend class ProcessGroupXCCL;
  };

  struct Options : Backend::Options {
    // NOTE: timeout in ProcessGroupNCCL::Options denote the timeout for
    // operations. This is only used when blockingWait_ is enabled.
    explicit Options(bool is_high_priority_stream = false);

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false) {
      return c10::make_intrusive<Options>(is_high_priority_stream);
    }

    // Schedule XCCL operations on high priority CUDA streams
    bool is_high_priority_stream;

    std::shared_ptr<ProcessGroupXCCL> split_from;
    int64_t split_color{0};
    std::vector<uint64_t> global_ranks_in_group;
    std::string group_name;
  };

  class CollectivesImpl {
    public:
         static void allreduceImpl(at::Tensor& input, at::Tensor& output,
                        cclComm_t comm, c10::Stream& stream, const AllreduceOptions& opts);

  };

  // If you wish to create multiple process groups, each with a potentially
  // different rank and size, you can do so by passing a new store instance
  // to each one. If you have only a single store object, you can
  // use the `c10d::PrefixStore` to derive scoped instances.
  // This is also what the Python API in torch.distributed does.
  //
  // The process group instance keeps a reference to the store because
  // it may be used long after the constructor runs. In fact, the constructor
  // doesn't create any NCCL communicators. A single NCCL communicator can
  // only be used on a specific set of devices, and are therefore created
  // on-demand when a collective runs. If another collective is executed later,
  // against a different set of devices, the process group creates another NCCL
  // communicator. These NCCL communicators are cached and reused if possible.
  //
  ProcessGroupXCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  // This constructor includes the deprecated `groupName` argument.
  // If you have existing code that uses the `groupName`, you can replace
  // it by specifying a `c10d::PrefixStore(groupName, store)` for store.
  C10_DEPRECATED ProcessGroupXCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      const std::string& groupName,
      c10::intrusive_ptr<Options> options = Options::create())
      : ProcessGroupXCCL(store, rank, size, options) {}
  ~ProcessGroupXCCL() override;

  // This function returns a local uid for ProcessGroupXCCL.
  uint64_t getUid() {
    return static_cast<uint64_t>(local_id_);
  }

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  const std::string getBackendName() const override {
    return std::string(XCCL_BACKEND_NAME);
  }

  bool supportsSplitting() const override {
    return false;
  }

  void startCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing() override;

  // For specifying a composite optype, such as ALLGATHER and REDUCE_SCATTER
  c10::intrusive_ptr<Work> endCoalescing(OpType optype);

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> _broadcast_oop(
      at::Tensor& outputTensors,
      at::Tensor& inputTensors,
      const BroadcastOptions& opts = BroadcastOptions());

  c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> _reduce_oop(
      at::Tensor& outputTensors,
      at::Tensor& inputTensors,
      const ReduceOptions& opts = ReduceOptions());

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  void groupStart();

  void groupEnd();

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  // Unsupported Ops
  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  // Agrees on an initial sequence number for the whole group by having rank 0
  // create it and broadcast it to other ranks using the store.
  void setSequenceNumberForGroup() override;

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  uint64_t getSequenceNumberForGroup() override;

  void waitForPendingWorks() override;

  void enableCollectivesTiming() override;

  void eagerConnectSingleDevice(at::Device device) override;

  // Helper that encapsulates work shared across all collective communication
  // primitives.  The callbacks have the following signatures:
  //
  //    ncclResult_t fn(at::Tensor& input, at::Tensor& output,
  //                    cclComm_t, at::cuda::CUDAStream&);
  //    void {pre,post}(std::vector<at::cuda::CUDAStream&>);
  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      OpType opType,
      const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false,
      bool nanCheck = true) {
      return collective<Fn>(
      {input},
      {output},
      fn,
      [](c10::Stream&,
         c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {},
      [](c10::Stream&,
         c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {},
      opType);
      }

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType,
      const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false,
      bool nanCheck = true) {
      return collective<Fn>(
          {input},
          {output},
          fn,
          [](c10::Stream&,
             c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {},
          [](c10::Stream&,
             c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {},
          opType);
      }

  template <typename Fn, typename PreProcess, typename PostProcess>
   c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType,
      const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false,
      bool nanCheck = true);

  template <typename Fn>
   c10::intrusive_ptr<Work> collectiveCoalesced(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      OpType opType,
      const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false);

  // Helper that encapsulates work shared across point-to-point communication
  // primitives. It is the same structure as the helper used for collective
  // communication primitives.
  template <typename Fn>
  c10::intrusive_ptr<Work> pointToPoint(
      at::Tensor& tensor,
      Fn fn,
      int peer,
      OpType opType,
      const char* profilingTitle = nullptr) {
      return collective<Fn>(
          tensor,
          fn,
          peer,
          opType,
          [](c10::Stream&,
             c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {},
          [](c10::Stream&,
             c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {},
          opType, profilingTitle);
   }

  template <typename Fn, typename PreProcess, typename PostProcess>
   c10::intrusive_ptr<Work> pointToPoint(
      at::Tensor& tensor,
      Fn fn,
      int peer,
      OpType opType,
      PreProcess pre,
      PostProcess post,
      const char* profilingTitle = nullptr);

  // Util function to assign timeout to each work.
  void assignTimeoutToWork(
      const c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work,
      const c10::intrusive_ptr<Options>& option);

  // Generates a prefix that is unique to this process group and rank, for
  // disambiguating logs
  std::string createLogPrefix() const;

  // Returns the unique prefix created in createLogPrefix
  const std::string& logPrefix() const;

    // Returns the global rank of the device. This function assumes that users
  // always create a default global process group(PG) which includes all
  // devices. It is called in the constructor of ProcessGroupNCCL, so it always
  // return the rank_ of the the very first PG created, aka, default global PG.
  const int& globalRank() const;

  // Returns the global ranks of a PG.
  const std::vector<uint64_t>& groupRanks() const;

  bool complexViewAsRealAllowed(const ReduceOp reduceOp);

  // Ensure thaht if record is True, the work obj will be enqueued via
  // workEnqueue
  virtual c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> initWork(
      at::Device& device,
      int rank,
      OpType opType,
      const char* profilingTitle = nullptr,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {},
      bool record = false);

  bool abort(std::optional<std::string> abortReason = std::nullopt);
  void shutdown(std::optional<std::string> reason = std::nullopt);

 protected:
  // In the timeout case and we will dump debug info such as the NCCL flight
  // recorder to storage. Down the road, if we have more complicated or blocking
  // operations, we might need to use a side thread to do it.
  bool dumpDebuggingInfo();

 private:
  int globalRankStart;
  int globalRankStride;

 protected:
  // A helper function to wait for a future to complete or timeout.
  void waitForFutureOrTimeout(
      std::future<bool>& fut,
      const std::chrono::milliseconds& timeOutMilSec,
      const std::string& futDescription,
      bool throwException = false,
      bool log = false);


  // The store is used to broadcast the NCCL unique ID of rank 0. This store
  // comes with prefix and it is different across ProcessGroup NCCL instances
  // (aka, different ProcessGroups).
  c10::intrusive_ptr<Store> store_;

  const c10::intrusive_ptr<Options> options_;

  // The number of NCCL communicators that have been created during
  // the lifetime of this process group. This sequence number is
  // used to scope keys used in the store.
  uint64_t ncclCommCounter_{0};

  // Mutex to guard maps like devNCCLCommMap_.
  std::mutex mutex_;

  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminateProcessGroup_;

  // Whether or not we should terminate the heartbeat monitoring threads.
  std::atomic<bool> terminateHeartbeatMonitorThread_;

  // Whether we are in the shutdown mode when we are trying to get debug info,
  // such as desync report.
  std::atomic<bool> collectiveDebugInfoMode_;

  // Whether there are hooks pending to be fired
  std::atomic<bool> hasPendingHooks_;

  // This is the signal from watchdog threads to indicate whether the monitor
  // thread should dump. Making it static so that it is accessiable from all the
  // PGs. With this flag, monitor thread would dump debug info under any one of
  // the three conditions:
  //
  // 1: watchdog thread of any PG detects a collective timeout.
  // 2: timeout signal is received from other ranks through tcpstore.
  // 3: current PG's watchdog heartbeat timeout occurs.
  //
  // Note that only the monitor thread from PG0 will dump the debug info for
  // case one and two so that the debug info is only dumped once.
  static std::atomic<bool> shouldDump_;

  // Mutex to Guard workMetaList_
  std::mutex workMetaListMutex_;

  // Mutex to Guard monitorWakeUpCV_
  std::mutex monitorMutex_;

  bool writeDebugInfo_ = false;

  // Whether or not to create start CUDAEvent and enable timing for start
  // and end events. Note that enableTiming_ is always true if desyncDebug_
  // is set to true.
  std::atomic<bool> enableTiming_;

  // Condition Variable for watchdog thread sleep
  std::condition_variable workMetaListCV_;

  // Condition Variable for monitor thread to wake up early
  std::condition_variable monitorWakeUpCV_;

  // Vector to Store WorkXCCL pointers
  std::list<ProcessGroupXCCL::WorkXCCL> workMetaList_;

  std::chrono::time_point<std::chrono::steady_clock> lastWorkListUpdateTime_;

  // Mutex to Guard workMetaList_
  std::mutex completedWorkListMutex_;

  // Condition Variable for watchdog thread sleep
  std::condition_variable completedWorkListCV_;

  std::list<ProcessGroupXCCL::WorkXCCL> completedWorkList_;

  // Add Work Pointer to workVector
  void workEnqueue(c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>);

  // The CUDA streams used by NCCL kernels
  std::unordered_map<std::string, c10::xpu::XPUStream> ncclStreams_;

  // The CUDA events used to sync NCCL streams
  std::unordered_map<std::string, c10::Event> ncclEvents_;

  // Device Indexes used for all collectives in this group
  std::set<int> usedDeviceIdxs_;

  // Flag to denote if a coalescing groupStart/groupEnd block is active
  int coalescing_state_ = 0;

  // Stores device indexes for all collectives run inside a coalescing block
  at::Device coalescedDevice_ = at::Device("xpu");

  // Stores communicators for all collectives run inside a coalescing block
  std::shared_ptr<void*> coalescedComm_ = nullptr;

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
  bool blockingWait_ = false;

  // Whether or not TORCH_NCCL_AVOID_RECORD_STREAMS was set
  bool avoidRecordStreams_ = false;

  // The number of active ncclGroupStart() calls. This counter will be increased
  // by 1 when ncclGroupStart() is called and decreased by 1 when ncclGroupEnd()
  // is called.
  static thread_local uint64_t ncclActiveGroupCounter_;

  // Counting for the sequential number of NCCL collective call.
  // (specifically, how many actual kernels we launched, which differs from
  // op_id_ when coalescing is enabled)
  uint64_t seqCollective_{0};

  // Counting for the sequential number of NCCL P2P calls.
  uint64_t seqP2P_{0};

  // Incrementing counter for logical operations (collective or p2p) issued on
  // the ProcessGroup
  uint64_t op_id_{0};

  // The number of ProcessGroupXCCL created on the current rank.
  size_t local_id_;

  std::string logPrefix_;

  // Number of devices on this node.
  int localDeviceCount_{0};
};
} // namespace c10d

#endif // USE_C10D_NCCL
