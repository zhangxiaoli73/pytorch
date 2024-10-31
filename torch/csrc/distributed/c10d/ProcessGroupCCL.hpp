#pragma once

#ifdef USE_C10D_XCCL

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <ATen/DynamicLibrary.h>
#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>
#include <torch/custom_class.h>

namespace c10d {

typedef void* cclComm_t;

enum ErrorHandlingMode {
  NoHandling = 0,
  TearDown = 1,
  CleanUpOnly = 2,
  SkipCleanUp = 3
};

#define SHOULD_CLEAN_UP(a) (a != NoHandling && a != SkipCleanUp)

#define SHOULD_TEAR_DOWN(a) (a != NoHandling && a != CleanUpOnly)

class TORCH_API ProcessGroupCCL : public Backend {
 public:
  class WorkCCL : public Work, public std::enable_shared_from_this<WorkCCL> {
   public:
    friend struct WorkInfo;

    // Constructor takes a list of CUDA devices
    WorkCCL(
        const std::string& pgUID,
        const std::string& pgDesc,
        at::Device& device,
        int rank,
        OpType opType,
        uint64_t seq,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputs = std::nullopt,
        bool desyncDebug = false,
        bool enableTiming = false,
        bool cudaEventCacheEnabled = false,
        DebugLevel distDebugLevel = DebugLevel::Off);
    // Copy constructor doing partial copy without outputs_. Cleanup thread
    // monitors and removes finished works. However it will deadlock when
    // destructs outputs_ tensors who are view tensors in autograd graph.
    WorkCCL(const WorkCCL& w);

    ~WorkCCL() override;

    // Checks if the CCL kernel has started to execute.
    bool isStarted();

    // Checks if request has completed. In this specific case of CCL, it checks
    // if the CCL operation has completed on the GPU in its own CCL stream.
    // Non-blocking operation.
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for CCL work.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void abort() override;

    // Let current stream wait on the completing of the CCL work
    // Throws on exceptions. Blocking operation, which will wait for work
    // completion.
    void synchronize() override;

    // Synchronize streams by blocking each on the CCL stream
    void synchronizeStream();

    // Helper function to handle exception (throw if needed).
    void handleException(ErrorHandlingMode asyncErrorHandling);

    // Helper function that checks if the CCL kernels have finished
    // execution on the GPUs
    bool finishedGPUExecution();

    // Get a Future object that will be marked as completed internally.
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    float getDuration() const override;

    uint64_t getSequencenumber() const override;

    const std::string& logPrefix() const;

    // Helper function that sets an exception_ptr on the WorkCCL object.
    void setException(std::exception_ptr exception_ptr);

    // Helper function that returns True if the WorkCCL object has timed out
    // and False otherwise.
    // In case of timeout, set exception on the WorkCCL object.
    bool checkTimeout(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt);

//    std::vector<at::Tensor> result() override;

     // todo: zl_debug protected or public?
//   protected:
    // The process group unique id
    std::string pgUID_;

    // The process group description
    std::string pgDesc_;

    // The cached list of CUDA devices to operate on
    at::Device device_;

    // The start CUDA event of CCL operator tracking this work item. These
    // start CUDA events are needed by desync debugging if enabled.
    std::shared_ptr<c10::Event> cclStartEvent_;

    // The end CUDA event of CCL operator tracking this work item.
    std::shared_ptr<c10::Event> cclEndEvent_;

    // The CCL communicator used for this work item.
    std::shared_ptr<cclComm_t> cclComm_;

    // Tensors used for barrier op
    at::Tensor barrierTensor_;

    // Clone of blockingWait_ from ProcessGroupCCL.
    bool blockingWait_ = false;

    // Clone of avoidRecordStreams_ from ProcessGroupCCL.
    bool avoidRecordStreams_ = false;

    // Clone of opTimeout_ from ProcessGroupCCL.
    std::chrono::milliseconds opTimeout_;

    // Ephemeral timeouts are owned by exactly one work,
    // and reset after that work completes.
    // There may be more than one ephemeral timeout active at the same time,
    // and this variable is used to track the ownership of ephemeral timeout.
    std::chrono::milliseconds ownedEphermeralTimeout_ =
        std::chrono::milliseconds(0);

    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    // Record the collective sequential number.
    uint64_t seq_;

    // Indicates if the CCL start event has been updated to the store trace.
    // This will be used by desync debug.
    bool startTraceUpdated_{false};

    // Record collective sizes for debug. We only record the size on the first
    // device as multi-device per process is deprecated
    size_t numelIn_ = -1;
    size_t numelOut_ = -1;

    bool cclCommIsAborted = false;

    // Wrapper method for the static checkForCCLErrors which can be overridden
    // for tests.
    virtual std::exception_ptr checkForCCLErrors();

    friend std::ostream& operator<<(
        std::ostream& output,
        const WorkCCL& workCCL);

    // Helper function for synchronize
    void synchronizeInternal(std::chrono::milliseconds timeout);

    // Checks for CCL errors and sets an appropriate exception_ptr.
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

    // Store a reference to CCL collective's outputs, used by result and to
    // give a more descriptive message when representing the Work as a string.
    std::shared_ptr<std::vector<at::Tensor>> outputs_;

    // TORCH_CCL_AVOID_RECORD_STREAMS implementation helper.
    // Stores references to participating non-output tensors (ie inputs,
    // flattened intermediates).
    // We'll clear this list in synchronizeStream, just after user-facing
    // stream(s) are synced with the CCL work stream(s).
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
    friend class ProcessGroupCCL;
  };

//  class AccEventCache {
//   public:
//    AccEventCache();
//    std::shared_ptr<c10::Event> create(bool timing);
//    static AccEventCache& get();
//
//   private:
//    std::mutex cacheMutex_;
//    // NOTE: We intentionaly store raw pointers so that
//    // we do not attempt to destroy the event objects on process exit,
//    // because cuda may be gone.
//    std::vector<c10::Event*>
//        eventsArray_[2]; // 0 for timing=false, 1 for timing=true
//  };

   // todo: zl_debug how to avoid such static?
//   static bool cclCommIsAborted;
   static void cclCommAbort(std::shared_ptr<cclComm_t> cclComm_);
   static std::exception_ptr checkForCCLErrorsInternal(std::shared_ptr<cclComm_t> cclComm_);

   ProcessGroupCCL(int rank, int size);
   ~ProcessGroupCCL() override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

//  c10::intrusive_ptr<Work> allreduce_coalesced(
//      std::vector<at::Tensor>& tensors,
//      const AllreduceCoalescedOptions& opts =
//          AllreduceCoalescedOptions()) override;
//
//  c10::intrusive_ptr<Work> reduce(
//      std::vector<at::Tensor>& tensors,
//      const ReduceOptions& opts = ReduceOptions()) override;
//
//
//  c10::intrusive_ptr<Work> allgather(
//      std::vector<std::vector<at::Tensor>>& outputTensors,
//      std::vector<at::Tensor>& inputTensors,
//      const AllgatherOptions& opts = AllgatherOptions()) override;
//
//  c10::intrusive_ptr<Work> _allgather_base(
//      at::Tensor& outputbuffer,
//      at::Tensor& inputbuffer,
//      const AllgatherOptions& opts = AllgatherOptions()) override;
//
//  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
//      std::vector<at::Tensor>& outputs,
//      std::vector<at::Tensor>& inputs,
//      const AllgatherOptions& opts = AllgatherOptions()) override;
//
//  c10::intrusive_ptr<Work> reduce_scatter(
//      std::vector<at::Tensor>& outputTensors,
//      std::vector<std::vector<at::Tensor>>& inputTensors,
//      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;
//
//  c10::intrusive_ptr<Work> _reduce_scatter_base(
//      at::Tensor& outputTensor,
//      at::Tensor& inputTensor,
//      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;
//
//  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
//      std::vector<at::Tensor>& outputs,
//      std::vector<at::Tensor>& inputs,
//      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;
//
//  c10::intrusive_ptr<Work> barrier(
//      const BarrierOptions& opts = BarrierOptions()) override;
//
//  c10::intrusive_ptr<Work> alltoall_base(
//      at::Tensor& outputTensor,
//      at::Tensor& inputTensor,
//      std::vector<int64_t>& outputSplitSizes,
//      std::vector<int64_t>& inputSplitSizes,
//      const AllToAllOptions& opts = AllToAllOptions()) override;
//
//  c10::intrusive_ptr<Work> alltoall(
//      std::vector<at::Tensor>& outputTensors,
//      std::vector<at::Tensor>& inputTensors,
//      const AllToAllOptions& opts = AllToAllOptions()) override;
//
//  c10::intrusive_ptr<Work> send(
//      std::vector<at::Tensor>& tensors,
//      int dstRank,
//      int tag) override;
//
//  c10::intrusive_ptr<Work> recv(
//      std::vector<at::Tensor>& tensors,
//      int srcRank,
//      int tag) override;
//
//  c10::intrusive_ptr<Work> gather(
//      std::vector<std::vector<at::Tensor>>& outputTensors,
//      std::vector<at::Tensor>& inputTensors,
//      const GatherOptions& opts = GatherOptions()) override;
//
//  c10::intrusive_ptr<Work> scatter(
//      std::vector<at::Tensor>& outputTensors,
//      std::vector<std::vector<at::Tensor>>& inputTensors,
//      const ScatterOptions& opts = ScatterOptions()) override;

  uint64_t getSequenceNumberForGroup() override {
     return seqCollective_;
   }

  template <typename Fn, typename T>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      T opts,
      OpType opType);

  c10::Stream getCCLStream(const std::string& deviceKey,at::Device& device);
  virtual void allreduce_impl(at::Tensor& input, at::Tensor& output,
      const AllreduceOptions& opts, c10::Stream stream, OpType opType) {
      TORCH_CHECK(false,
        c10::str(
            "Backend ", getBackendName(), " does not implement endCoalescing"));
  }

  virtual void broadcast_impl(at::Tensor& input, at::Tensor& output,
      const BroadcastOptions& opts, c10::Stream stream, OpType opType) {
      TORCH_CHECK( false,
        c10::str(
            "Backend ", getBackendName(), " does not implement endCoalescing"));
  }


  virtual c10::intrusive_ptr<ProcessGroupCCL::WorkCCL> initCCLWork(
      at::Device& device,
      int rank,
      OpType opType,
      const char* profilingTitle = nullptr,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {});

  protected:
      uint64_t seqCollective_{0};
      std::unordered_map<std::string, c10::Stream> cclStreamsMap_;
      std::unordered_map<std::string, c10::Event> cclEventsMap_;

      // Mutex to guard maps like devNCCLCommMap_.
      std::mutex mutex_;
};

} // namespace c10d

#endif // USE_C10D_XCCL
