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

void setXCCLEnvVar(std::string envVarName, int val) {
  setenv(envVarName.c_str(), std::to_string(val).c_str(), val);
}

void setXCCLEnvVar(std::string envVarName, std::string val) {
  setenv(envVarName.c_str(), val.c_str(), 1);
}

bool with_mpirun() {
  return (getenv("MPI_LOCALRANKID") || getenv("MPI_LOCALNRANKS") ||
          getenv("PMI_RANK") || getenv("PMI_SIZE") || getenv("PMIX_RANK"))
      ? true
      : false;
}

struct AutoXcclGroup {
  AutoXcclGroup();
  ~AutoXcclGroup() noexcept(false);
};
} // namespace

static std::vector<std::string> TORCH_XCCL_BLOCKING_WAIT = {
    "TORCH_XCCL_BLOCKING_WAIT",
    "XCCL_BLOCKING_WAIT"};

using xcclComm_t = ccl::communicator;
using XCCL_KVS = ccl::shared_ptr_class<ccl::kvs>;
constexpr const char* XCCL_BACKEND_NAME = "xccl";

class TORCH_API ProcessGroupXCCL : public Backend {
 public:
  class WorkXCCL : public Work {
   public:
    WorkXCCL(
        at::Device& device,
        int rank,
        OpType opType,
        const std::optional<std::vector<at::Tensor>>& inputs = std::nullopt);
    WorkXCCL(const WorkXCCL& w);
    ~WorkXCCL() override;

    bool isCompleted() override;

    bool isSuccess() const override {
      TORCH_CHECK(
          false, "ProcessGroupXCCL::WorkXCCL::isSuccess not implemented");
    }

    void abort() override {
      TORCH_CHECK(false, "ProcessGroupXCCL::WorkXCCL::abort not implemented");
    }

    void synchronize() override;

    void synchronizeStream();

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
      return future_;
    }

    std::vector<at::Tensor> result() override {
      return *outputs_;
    }

    bool checkTimeout(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt);

   protected:
    at::Device device_;
    std::shared_ptr<at::xpu::XPUEvent> xcclEndEvent_;
    at::Tensor barrierTensor_;
    bool blockingWait_ = false;
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

   private:
    void synchronizeInternal(std::chrono::milliseconds timeout);
    std::shared_ptr<std::vector<at::Tensor>> outputs_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
    friend class ProcessGroupXCCL;
  };

  ProcessGroupXCCL(const c10::intrusive_ptr<Store>& store, int rank, int size);

  C10_DEPRECATED ProcessGroupXCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      const std::string& groupName)
      : ProcessGroupXCCL(store, rank, size) {}

  ~ProcessGroupXCCL() override;

  const std::string getBackendName() const override {
    return std::string(XCCL_BACKEND_NAME);
  }

  void startCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing(OpType optype);

  std::shared_ptr<xcclComm_t> getXCCLComm(
      const std::string& deviceKey,
      at::Device& device);

  virtual c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> initWork(
      at::Device& device,
      int rank,
      OpType opType,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {});

  template <typename Fn, typename input_t, typename output_t>
  c10::intrusive_ptr<Work> collective(
      input_t& input,
      output_t& output,
      Fn fn,
      OpType opType);

  template <
      typename Fn,
      typename input_t,
      typename output_t,
      typename PreProcess,
      typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      input_t& input,
      output_t& output,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType);

  template <
      typename Fn,
      typename input_t,
      typename output_t,
      typename PreProcess,
      typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      std::vector<input_t>& inputs,
      std::vector<output_t>& outputs,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType);

  template <typename Fn>
  c10::intrusive_ptr<Work> collectiveCoalesced(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      OpType opType);

  c10::intrusive_ptr<Work> allreduce_impl(
      at::Tensor& tensor,
      const AllreduceOptions& opts = AllreduceOptions());

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

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

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
      const AllgatherOptions& opts = AllgatherOptions()) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::allgather_coalesced not implemented");
  }

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
      int tag) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::send not implemented");
  }

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    TORCH_CHECK(false, "ProcessGroupXCCL::recv not implemented");
  }

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

 protected:
  std::unordered_map<std::string, at::xpu::XPUStream> xcclStreams_;
  std::unordered_map<std::string, std::shared_ptr<xcclComm_t>>
      inInitializationCommMap_;
  std::unordered_map<std::string, std::shared_ptr<xcclComm_t>> devXCCLCommMap_;
  c10::intrusive_ptr<Store> store_;
  std::mutex mutex_;
  std::set<int> usedDeviceIdxs_;
  int coalescing_state_ = 0;
  at::Device coalescedDevice_ = at::Device("xpu");
  std::shared_ptr<xcclComm_t> coalescedComm_ = nullptr;
  bool blockingWait_ = false;
  static thread_local uint64_t xcclActiveGroupCounter_;
};
} // namespace c10d

#endif // USE_C10D_XCCL
