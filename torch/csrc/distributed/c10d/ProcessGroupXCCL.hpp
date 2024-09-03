#pragma once

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#ifdef USE_C10D_XCCL

#include <oneapi/ccl.hpp>
#include <torch/csrc/xpu/xccl.h>
#include <torch/csrc/xpu/Stream.h>
#include <torch/csrc/xpu/Event.h>
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

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {

constexpr const char* XCCL_BACKEND_NAME = "xccl";
using namespace torch::xpu::xccl;

class ProcessGroupXCCL : public Backend {
 public:
  class WorkXCCL : public Work {
   public:
    WorkXCCL(
        at::Device& device,
        int rank,
        OpType opType,
        const std::optional<std::vector<at::Tensor>>& inputs = std::nullopt);
    // WorkXCCL(
    //     std::vector<std::vector<at::Tensor>> outputTensors,
    //     int rank = -1,
    //     OpType opType = OpType::UNKNOWN,
    //     const c10::optional<std::vector<at::Tensor>>& inputTensors =
    //         c10::nullopt)
    //     : Work(rank, opType), outputTensors_(std::move(outputTensors)) {}
    WorkXCCL(const WorkXCCL& w);
    // ~WorkXCCL() override {
    //   // Ensures all events are properly handled before destruction
    //   for (auto& event : events_) {
    //     event.wait();
    //   }
    // }
    ~WorkXCCL() override;
    bool isCompleted() override {
      TORCH_CHECK(
          false, "ProcessGroupXCCL::WorkXCCL::isCompleted not implemented");
      // for (auto& event : events_) {
      //   if (!event.test()) {
      //     return false;
      //   }
      // }
      // return true;
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
    // void wait() {
    //   std::unique_lock<std::timed_mutex> lock(mutex_);
    //   for (auto& event : events_) {
    //     event.wait();
    //   }
    //   events_.clear();
    // }

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
      return future_;
    }

    std::vector<at::Tensor> result() override {
      TORCH_CHECK(false, "ProcessGroupXCCL::WorkXCCL::result not implemented");
      // return outputTensors_.empty() ? std::vector<at::Tensor>()
      //                               : outputTensors_[0];
    }

   protected:
    at::Device device_;
    std::shared_ptr<at::xpu::XPUEvent> xcclEndEvent_;
    // std::vector<ccl::event> events_;
    // std::shared_ptr<xcclComm_t> xcclComm_;
    // const std::vector<std::vector<at::Tensor>> outputTensors_;
   private:
    std::shared_ptr<std::vector<at::Tensor>> outputs_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
    friend class ProcessGroupXCCL;
  };

  explicit ProcessGroupXCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size)
      : store_(store), Backend(rank, size) {}

  ~ProcessGroupXCCL() override;

  const std::string getBackendName() const override {
    return std::string(XCCL_BACKEND_NAME);
  }

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
