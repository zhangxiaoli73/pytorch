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
#include <torch/csrc/distributed/c10d/ProcessGroupGCCL.hpp>
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

static std::vector<std::string> TORCH_XCCL_BLOCKING_WAIT = {
    "TORCH_XCCL_BLOCKING_WAIT",
    "XCCL_BLOCKING_WAIT"};

using xcclComm_t = ccl::communicator;
using XCCL_KVS = ccl::shared_ptr_class<ccl::kvs>;
constexpr const char* XCCL_BACKEND_NAME = "xccl";

class TORCH_API ProcessGroupXCCL : public ProcessGroupGCCL {
 public:
  class WorkXCCL : public WorkGCCL {
   public:
    WorkXCCL(
        at::Device& device,
        int rank,
        OpType opType,
        const std::optional<std::vector<at::Tensor>>& inputs = std::nullopt);
    WorkXCCL(const WorkXCCL& w);
    ~WorkXCCL();

   protected:
    at::Device device_;
    std::shared_ptr<at::xpu::XPUEvent> xcclEndEvent_;
    bool blockingWait_ = false;
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

   private:
    void synchronizeInternal(std::chrono::milliseconds timeout);
    std::shared_ptr<std::vector<at::Tensor>> outputs_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
    friend class ProcessGroupXCCL;
  };

  class CollectivesXCCLImpl: public CollectivesImpl {
    public:
         ncclResult_t allreduceImpl(at::Tensor& input, at::Tensor& output, ncclComm_t comm, c10::Stream& stream) override;
         ncclResult_t broadcastImpl(at::Tensor& input, at::Tensor& output, ncclComm_t comm, c10::Stream& stream) override;
         ncclResult_t reduceImpl(at::Tensor& input, at::Tensor& output, ncclComm_t comm, c10::Stream& stream) override;
         ncclResult_t allgatherImpl(at::Tensor& input, at::Tensor& output, ncclComm_t comm, c10::Stream& stream) override;
         ncclResult_t reducescatterImpl(at::Tensor& input, at::Tensor& output, ncclComm_t comm, c10::Stream& stream) override;
         ncclResult_t all2allImpl(at::Tensor& input, at::Tensor& output, ncclComm_t comm, c10::Stream& stream) override;
         ncclResult_t sendImpl(at::Tensor& input, ncclComm_t comm, c10::Stream& stream, int dst) override;
         ncclResult_t recvImpl(at::Tensor& output, ncclComm_t comm, c10::Stream& stream, int dst) override;
         ncclResult_t gatherImpl(at::Tensor& output, ncclComm_t comm, c10::Stream& stream, int dst) override;
         ncclResult_t scatterImpl(at::Tensor& input, at::Tensor& output, ncclComm_t comm, c10::Stream& stream) override;
         ncclResult_t groupStartImpl(std::shared_ptr<NCCLComm> comm) override;
         ncclResult_t groupEndImpl() override;
         ncclResult_t dataTypes override;
         ncclResult_t reductionTypes override;
  }

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

  std::shared_ptr<xcclComm_t> getXCCLComm(
      const std::string& deviceKey,
      at::Device& device);

  virtual c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> initWork(
      at::Device& device,
      int rank,
      OpType opType,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {});

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
      bool nanCheck = true) override;

  template <typename Fn>
  c10::intrusive_ptr<Work> collectiveCoalesced(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      OpType opType,
      const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false) override;

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> pointToPoint(
      at::Tensor& tensor,
      Fn fn,
      int peer,
      OpType opType,
      PreProcess pre,
      PostProcess post,
      const char* profilingTitle = nullptr) override;

 protected:
  std::unordered_map<std::string, at::xpu::XPUStream> xcclStreams_;
  std::unordered_map<std::string, std::shared_ptr<xcclComm_t>>
      inInitializationCommMap_;
  std::unordered_map<std::string, std::shared_ptr<xcclComm_t>> devXCCLCommMap_;
  c10::intrusive_ptr<Store> store_;
  std::mutex mutex_;
  bool blockingWait_ = false;
};
} // namespace c10d

#endif // USE_C10D_XCCL
