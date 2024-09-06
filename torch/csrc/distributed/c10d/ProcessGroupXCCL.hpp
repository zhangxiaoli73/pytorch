#pragma once

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#ifdef USE_C10D_XCCL
#include <oneapi/ccl.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/xpu/Event.h>
#include <torch/csrc/xpu/Stream.h>
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
    bool isCompleted() override {
      TORCH_CHECK(
          false, "ProcessGroupXCCL::WorkXCCL::isCompleted not implemented");
    }

    bool isSuccess() const override {
      TORCH_CHECK(
          false, "ProcessGroupXCCL::WorkXCCL::isSuccess not implemented");
    }

    void abort() override {
      TORCH_CHECK(false, "ProcessGroupXCCL::WorkXCCL::abort not implemented");
    }

    void synchronize() override;

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
      return future_;
    }

    std::vector<at::Tensor> result() override {
      TORCH_CHECK(false, "ProcessGroupXCCL::WorkXCCL::result not implemented");
    }

   protected:
    at::Device device_;
    std::shared_ptr<at::xpu::XPUEvent> xcclEndEvent_;
   private:
    std::shared_ptr<std::vector<at::Tensor>> outputs_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
    friend class ProcessGroupXCCL;
  };

  explicit ProcessGroupXCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size)
      : Backend(rank, size), store_(store) {}

  ~ProcessGroupXCCL() override;

  const std::string getBackendName() const override {
    return std::string(XCCL_BACKEND_NAME);
  }

  std::shared_ptr<xcclComm_t> getXCCLComm(
      const std::string& deviceKey,
      at::Device& device);

  virtual c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> initWork(
      at::Device& device,
      int rank,
      OpType opType,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {});

  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      OpType opType);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType);

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  // c10::intrusive_ptr<Work> barrier(
  //     const BarrierOptions& opts = BarrierOptions()) override;

  static c10::intrusive_ptr<Backend> createProcessGroupXCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank = -1,
      int size = -1);

 public:
  std::unordered_map<std::string, at::xpu::XPUStream> xcclStreams_;
  std::unordered_map<std::string, std::shared_ptr<xcclComm_t>>
      inInitializationCommMap_;
  std::unordered_map<std::string, std::shared_ptr<xcclComm_t>> devXCCLCommMap_;
  c10::intrusive_ptr<Store> store_;
  std::mutex mutex_;
};

} // namespace c10d

#endif // USE_C10D_XCCL
