#ifdef USE_C10D_XCCL

#include <torch/csrc/distributed/c10d/ProcessGroupXCCL.hpp>
#include <fstream>
#include <comm/XPUGuard.h>
#include <exception>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <ATen/detail/FunctionTraits.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>

namespace c10d {

namespace {

// wait nonblocking implement
AutoXcclGroup::AutoXcclGroup() {
  ccl::group_start();
}

AutoXcclGroup::~AutoXcclGroup() noexcept(false) {
  ccl::group_end();
}

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
    // use for allgather
    {at::kFloat8_e5m2, ccl::datatype::uint8},
    {at::kFloat8_e4m3fn, ccl::datatype::uint8},
    {at::kFloat8_e4m3fnuz, ccl::datatype::uint8},
    {at::kFloat8_e5m2fnuz, ccl::datatype::uint8},
};

bool computeLengthsAndCheckAndGetFlat(
    const std::vector<at::Tensor>& tensors,
    std::vector<size_t>& lengths,
    at::Tensor& flatTensor,
    int64_t& flatLength) {
  int64_t groupSize = tensors.size();
  auto firstTensor = tensors[0];
  int64_t totalSize = 0;
  bool isFlat = true;

  auto storage = firstTensor.storage();
  int64_t firstStorageOffset = firstTensor.storage_offset();

  for (int i = 0; i < groupSize; i++) {
    auto& curTensor = tensors[i];
    int64_t length = curTensor.numel();
    lengths[i] = length;
    totalSize += length;

    if (isFlat &&
        (!storage.is_alias_of(curTensor.storage()) ||
         curTensor.storage_offset() !=
             firstStorageOffset + totalSize - length)) {
      isFlat = false;
    }
  }

  flatLength = totalSize;

  if (isFlat) {
    flatTensor = firstTensor;
  } else {
    flatTensor = at::empty({totalSize}, firstTensor.options());
  }

  return isFlat;
}

bool check_same_size(const std::vector<at::Tensor>& input_tensors) {
  for (const auto& input_tensor : input_tensors) {
    if (!input_tensors[0].is_same_size(input_tensor)) {
      return false;
    }
  }
  return true;
}

void check_xpu_single_tensor(
    const at::Tensor& tensor,
    const bool p2p = false // whether operation is a P2P operation
) {
  if (!tensor.is_xpu() || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be XPU and dense");
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

int64_t check_xpu_tensors_same_device(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    C10_THROW_ERROR(ValueError, "Tensor list must be nonempty");
  }

  const auto& first = tensors.front();

  int64_t total_numel = 0;
  for (const auto& t : tensors) {
    if (!t.is_xpu() || t.is_sparse()) {
      C10_THROW_ERROR(ValueError, "Tensors must be XPU and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      C10_THROW_ERROR(TypeError, "Tensors must have identical type");
    }
    if (!t.is_non_overlapping_and_dense()) {
      C10_THROW_ERROR(ValueError, "Tensors must be non-overlapping and dense");
    }
    TORCH_CHECK_WITH(
        ValueError,
        t.get_device() == tensors[0].get_device(),
        "Expected list of tensors on the same device");
    total_numel += t.numel();
  }

  return total_numel;
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
    C10_THROW_ERROR(
        ValueError,
        "Cannot use ReduceOp." + reduce_op_to_string(reduceOp) + " with XCCL");
  }
}

void syncStream(
    at::Device& device,
    at::xpu::XPUEvent& xcclEvent,
    at::xpu::XPUStream& xcclStream) {
  xcclEvent.record(at::xpu::getCurrentXPUStream(device.index()));
  xcclEvent.block(xcclStream);
}

bool complexViewAsRealAllowed(const ReduceOp reduceOp) {
  switch (reduceOp) {
    case ReduceOp::SUM:
      return true;
    case ReduceOp::UNUSED:
      return true;
    default:
      return false;
  }
  return false;
}

} // namespace

static std::mutex xcclCommDevIdxMapMutex;
static std::unordered_map<std::shared_ptr<xcclComm_t>, int> xcclCommDevIdxMap;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;
thread_local uint64_t ProcessGroupXCCL::xcclActiveGroupCounter_ = 0;

ProcessGroupXCCL::WorkXCCL::WorkXCCL(
    at::Device& device,
    int rank,
    OpType opType,
    const std::optional<std::vector<at::Tensor>>& inputs)
    : Work(rank, opType, "profilingTitle", inputs),
      device_(device),
      workStartTime_(std::chrono::steady_clock::now()) {
  unsigned char enable_timing = 0;
  xcclEndEvent_ = std::make_shared<at::xpu::XPUEvent>(enable_timing);
}

ProcessGroupXCCL::WorkXCCL::WorkXCCL(const WorkXCCL& w)
    : Work(w.rank_, w.opType_),
      device_(w.device_),
      xcclEndEvent_(w.xcclEndEvent_),
      blockingWait_(w.blockingWait_),
      workStartTime_(w.workStartTime_) {}

ProcessGroupXCCL::WorkXCCL::~WorkXCCL() = default;

bool ProcessGroupXCCL::WorkXCCL::isCompleted() {
  if (xcclEndEvent_ && xcclEndEvent_->query()) {
    return true;
  }
  return false;
}

void ProcessGroupXCCL::WorkXCCL::synchronize() {
  synchronizeInternal(kNoTimeout);
}

void ProcessGroupXCCL::WorkXCCL::synchronizeInternal(
    std::chrono::milliseconds timeout) {
  auto currentStream = at::xpu::getCurrentXPUStream(device_.index());
  xcclEndEvent_->block(currentStream);
  if (blockingWait_) {
    while (!isCompleted()) {
      auto currentTimepoint = std::chrono::steady_clock::now();
      auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          currentTimepoint - workStartTime_);
      if (timeElapsed >= timeout) {
        std::string exceptionMsg = c10::str(
            "Work ran for ",
            timeElapsed.count(),
            " milliseconds before timing out.");
        TORCH_CHECK(false, exceptionMsg)
      }

      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
  }
  if (barrierTensor_.defined()) {
    auto currentStream = at::xpu::getCurrentXPUStream(device_.index());
    currentStream.synchronize();
  }
}

bool ProcessGroupXCCL::WorkXCCL::wait(std::chrono::milliseconds timeout) {
  synchronizeInternal(timeout);
  return true;
}

constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple";

ProcessGroupXCCL::ProcessGroupXCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : Backend(rank, size), store_(store) {
  blockingWait_ = getCvarBool(TORCH_XCCL_BLOCKING_WAIT, false);
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

ProcessGroupXCCL::~ProcessGroupXCCL() = default;

c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> ProcessGroupXCCL::initWork(
    at::Device& device,
    int rank,
    OpType opType,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs) {
  auto r = c10::make_intrusive<ProcessGroupXCCL::WorkXCCL>(
      device, rank, opType, std::optional<std::vector<at::Tensor>>(inputs));
  return r;
}

std::shared_ptr<xcclComm_t> ProcessGroupXCCL::getXCCLComm(
    const std::string& deviceKey,
    at::Device& device,
    OpType opType,
    int p2pRank,
    bool isSendRecvSelf) {
  if (deviceKey.empty()) {
    C10_THROW_ERROR(
        DistBackendError,
        "Not able to create/get the XCCL Communicator since "
        "the devices are empty ");
  }

  usedDeviceIdxs_.insert(device.index());

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (devXCCLCommMap_.find(deviceKey) != devXCCLCommMap_.end()) {
      return devXCCLCommMap_[deviceKey];
    }
  }

  std::shared_ptr<xcclComm_t> XCCLComm;
  XCCL_KVS kvs = get_kvs(rank_, *store_);

  bool batchP2P = xcclActiveGroupCounter_ > 0;
  bool singleP2POp = isP2POp(opType, batchP2P);

  at::xpu::OptionalXPUGuard gpuGuard(device);

  for (const auto i : c10::irange(xcclActiveGroupCounter_)) {
    (void)i;
    ccl::group_end();
  }

  int numRanks, rank;
  if (!singleP2POp) {
    numRanks = getSize();
    rank = getRank();
  } else if (isSendRecvSelf) {
    numRanks = 1;
    rank = 0;
  } else {
    numRanks = 2;
    rank = p2pRank;
  }

  c10::impl::VirtualGuardImpl impl(device.type());
  c10::Stream stream = impl.getStream(device);
  sycl::queue& q = c10::xpu::XPUStream(stream).queue();

  auto ctx = ccl::create_context(q.get_context());
  ccl::vector_class<ccl::pair_class<int, ccl::device>> devs_rank;
  devs_rank.emplace_back(rank, ccl::create_device(q.get_device()));

  auto comms = ccl::create_communicators(numRanks, devs_rank, ctx, kvs);
  XCCLComm = std::make_shared<xcclComm_t>(std::move(comms[0]));

  {
    std::lock_guard<std::mutex> lock(mutex_);
    inInitializationCommMap_.emplace(deviceKey, XCCLComm);
  }

  for (const auto i : c10::irange(xcclActiveGroupCounter_)) {
    (void)i;
    ccl::group_start();
  }

  xcclStreams_.emplace(deviceKey, std::move(stream));
  xcclEvents_.emplace(deviceKey, at::xpu::XPUEvent());

  auto it = inInitializationCommMap_.find(deviceKey);
  if (it != inInitializationCommMap_.end()) {
    devXCCLCommMap_.emplace(deviceKey, std::move(it->second));
    inInitializationCommMap_.erase(deviceKey);

    xcclCommDevIdxMapMutex.lock();
    xcclCommDevIdxMap.emplace(XCCLComm, device.index());
    xcclCommDevIdxMapMutex.unlock();
  }

  it = devXCCLCommMap_.find(deviceKey);
  TORCH_INTERNAL_ASSERT(
      it != devXCCLCommMap_.end(), "Communicators not populated in cache!");

  return it->second;
}

void ProcessGroupXCCL::groupStart() {
  ccl::group_start();
  ++xcclActiveGroupCounter_;
}

void ProcessGroupXCCL::groupEnd() {
  ccl::group_end();
  --xcclActiveGroupCounter_;
}

// TODO: wait p2p enable
static constexpr int CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;
void ProcessGroupXCCL::startCoalescing() {
  coalescedDevice_.set_index(-1);
  coalescedComm_ = nullptr;
  coalescing_state_ |= CoalActive;
  groupStart();
}

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

  auto comm = coalescedComm_;
  auto device = coalescedDevice_;

  const auto key = std::to_string(device.index());
  auto stream = xcclStreams_.at(key);

  auto work = initWork(device, rank_, optype);
  work->blockingWait_ = blockingWait_;

  groupEnd();

  work->xcclEndEvent_->record(stream);

  coalescing_state_ = 0;
  coalescedComm_ = nullptr;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::endCoalescing() {
  // Default OpType to COALESCED if not specified
  return endCoalescing(OpType::COALESCED);
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupXCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType) {
  using traits = function_traits<Fn>;
  using attr_t = typename traits::template arg<2>::type;
  attr_t attr = ccl::create_operation_attr<attr_t>();

  auto device = inputs[0].device();
  const auto key = std::to_string(device.index());
  auto comm = getXCCLComm(key, device, opType);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalColl;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = comm;
    } else {
      TORCH_CHECK(coalescedComm_ == comm, MULTI_DEVICE_ERROR_MSG);
    }
  }

  auto stream = xcclStreams_.at(key);
  syncStream(device, xcclEvents_[key], stream);

  c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;
  work = initWork(device, rank_, opType);

  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  at::xpu::OptionalXPUGuard gpuGuard(device);

  pre(stream, work);

  for (const auto& input : inputs) {
    c10::xpu::XPUCachingAllocator::recordStream(
        input.storage().data_ptr(), stream);
  }

  fn(inputs[0], outputs[0], attr, *comm, stream);

  post(stream, work);

  if (!coalescing_state_) {
    work->xcclEndEvent_->record(stream);
  }

  std::vector<c10::Stream> streams = {stream.unwrap()};
  c10::MultiStreamGuard streamGuard(streams);
  std::vector<at::Device> devices{device};
  work->future_ = c10::make_intrusive<at::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(at::IValue(*work->outputs_));
  work->blockingWait_ = blockingWait_;

  return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupXCCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType) {
  auto inputs = std::vector<at::Tensor>{input};
  auto outputs = std::vector<at::Tensor>{output};
  return collective(inputs, outputs, fn, pre, post, opType);
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupXCCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    OpType opType) {
  return collective<Fn>(
      input,
      output,
      fn,
      [](at::xpu::XPUStream&, c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {
      },
      [](at::xpu::XPUStream&, c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {
      },
      opType);
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupXCCL::collectiveCoalesced(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    OpType opType) {
  using traits = function_traits<Fn>;
  using attr_t = typename traits::template arg<2>::type;
  attr_t attr = ccl::create_operation_attr<attr_t>();

  auto device = inputs[0].device();
  const auto key = std::to_string(device.index());
  auto comm = getXCCLComm(key, device, opType);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalColl;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = comm;
    } else {
      TORCH_CHECK(coalescedComm_ == comm, MULTI_DEVICE_ERROR_MSG);
    }
  }

  auto stream = xcclStreams_.at(key);
  syncStream(device, xcclEvents_[key], stream);

  c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;
  work = initWork(device, rank_, opType);

  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  at::xpu::OptionalXPUGuard gpuGuard(device);

  {
    AutoXcclGroup xccl_group_guard;
    for (const auto i : c10::irange(inputs.size())) {
      c10::xpu::XPUCachingAllocator::recordStream(
          inputs[i].storage().data_ptr(), stream);
      fn(inputs[i], outputs[i], attr, *comm, stream);
    }
  }

  work->xcclEndEvent_->record(stream);

  std::vector<c10::Stream> streams = {stream.unwrap()};
  c10::MultiStreamGuard streamGuard(streams);
  std::vector<at::Device> devices{device};
  work->future_ = c10::make_intrusive<at::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(at::IValue(*work->outputs_));
  work->blockingWait_ = blockingWait_;

  return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupXCCL::pointToPoint(
    at::Tensor& tensor,
    Fn fn,
    int peer,
    OpType opType,
    PreProcess pre,
    PostProcess post) {
  using traits = function_traits<Fn>;
  using attr_t = typename traits::template arg<1>::type;
  attr_t attr = ccl::create_operation_attr<attr_t>();

  auto device = tensor.device();
  std::string key;
  int p2pRank = 0, p2pTargetRank = 0;
  bool isSendRecvSelf = false;

  bool batchP2P = xcclActiveGroupCounter_ > 0;
  if (batchP2P) {
    key = std::to_string(device.index());
    p2pRank = rank_;
    p2pTargetRank = peer;
  } else {
    int lowRank = rank_ < peer ? rank_ : peer;
    int highRank = rank_ < peer ? peer : rank_;
    key = std::to_string(lowRank) + ":" + std::to_string(highRank);
    p2pRank = rank_ <= peer ? 0 : 1;
    isSendRecvSelf = rank_ == peer;
    p2pTargetRank = isSendRecvSelf ? 0 : 1 - p2pRank;
  }

  auto comm = getXCCLComm(key, device, opType, p2pRank, isSendRecvSelf);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalP2P;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = comm;
    } else {
      TORCH_CHECK(coalescedComm_ == comm, MULTI_DEVICE_ERROR_MSG);
    }
  }

  auto stream = xcclStreams_.at(key);
  syncStream(device, xcclEvents_[key], stream);

  c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;
  if (!coalescing_state_) {
    work = initWork(device, rank_, opType);
    work->outputs_ = std::make_shared<std::vector<at::Tensor>>();
    work->outputs_->push_back(tensor);
  }

  at::xpu::OptionalXPUGuard gpuGuard(device);

  if (!coalescing_state_) {
    pre(stream, work);
  }

  c10::xpu::XPUCachingAllocator::recordStream(
      tensor.storage().data_ptr(), stream);

  fn(tensor, attr, *comm, stream, p2pTargetRank);

  if (!coalescing_state_) {
    post(stream);

    work->xcclEndEvent_->record(stream);
    work->blockingWait_ = blockingWait_;

    {
      std::vector<c10::Stream> streams = {stream.unwrap()};
      c10::MultiStreamGuard streamGuard(streams);
      std::vector<at::Device> devices{device};
      work->future_ = c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()), devices);
      work->future_->markCompleted(at::IValue(*work->outputs_));
    }
    return work;
  } else {
    return nullptr;
  }
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupXCCL::pointToPoint(
    at::Tensor& tensor,
    Fn fn,
    int peer,
    OpType opType) {
  return pointToPoint(
      tensor,
      fn,
      peer,
      opType,
      [](at::xpu::XPUStream&, c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {
      },
      [](at::xpu::XPUStream&) {});
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /* unused */) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  check_xpu_single_tensor(tensor, true);

  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& input,
          ccl::pt2pt_attr attr,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          int dst) {
        ccl::event ret_evt;
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ret_evt = ccl::send(
            input.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            dst,
            comm,
            ccl::create_stream(stream.queue()),
            attr);
        return ret_evt;
      },
      dstRank,
      OpType::SEND);
  return ret;
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /* unused */) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  check_xpu_single_tensor(tensor, true);

  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& output,
          ccl::pt2pt_attr attr,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          int src) {
        ccl::event ret_evt;
        auto xcclDataType = getXcclDataType(output.scalar_type());
        ret_evt = ccl::recv(
            output.data_ptr(),
            (size_t)output.numel(),
            xcclDataType,
            src,
            comm,
            ccl::create_stream(stream.queue()),
            attr);
        return ret_evt;
      },
      srcRank,
      OpType::RECV);
  return ret;
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupXCCL::gather: " + msg);
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

  auto inputs = std::vector<at::Tensor>{inputTensor};
  return collective(
      inputs,
      outputs, // just to fit the collective interface
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ccl::allgather_attr attr, // just to fit interface
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          for (auto output : outputs) {
            c10::xpu::XPUCachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
        }
        {
          ccl::event ret_evt;
          auto xcclDataType = getXcclDataType(inputTensor.scalar_type());
          if (rank_ == root) {
            for (const auto r : c10::irange(size_)) {
              if (r != root) {
                // do receive
                ret_evt = ccl::recv(
                    outputs[r].data_ptr(),
                    (size_t)inputTensor.numel(),
                    xcclDataType,
                    r,
                    comm,
                    ccl::create_stream(stream.queue()));
              } else {
                // on its own rank, simply copy from the input
                outputs[r].copy_(inputTensor);
              }
            }
          } else {
            // do send
            ret_evt = ccl::send(
                inputTensor.data_ptr(),
                (size_t)inputTensor.numel(),
                xcclDataType,
                root,
                comm,
                ccl::create_stream(stream.queue()));
          }
          return ret_evt;
        }
      },
      [](at::xpu::XPUStream&, c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {
      },
      [](at::xpu::XPUStream&, c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {
      },
      OpType::GATHER);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupXCCL::scatter: " + msg);
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

  const auto root = opts.rootRank;

  auto outputs = std::vector<at::Tensor>{outputTensor};
  return collective(
      outputs,
      inputs, // just to fit the collective interface
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ccl::allgather_attr attr, // just to fit interface
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        if (getRank() == root) {
          for (auto input : inputs) {
            c10::xpu::XPUCachingAllocator::recordStream(
                input.storage().data_ptr(), stream);
          }
        }
        {
          ccl::event ret_evt;
          if (rank_ == root) {
            for (const auto r : c10::irange(size_)) {
              if (r != root) {
                // do send
                size_t send_count = inputs[r].numel();
                auto send_type = getXcclDataType(inputs[r].scalar_type());
                ret_evt = ccl::send(
                    inputs[r].data_ptr(),
                    send_count,
                    send_type,
                    r,
                    comm,
                    ccl::create_stream(stream.queue()));
              } else {
                // on its own rank, simply copy from the input
                outputTensor.copy_(inputs[r]);
              }
            }
          } else {
            // do receive
            size_t recv_count = outputTensor.numel();
            auto recv_type = getXcclDataType(outputTensor.scalar_type());
            ret_evt = ccl::recv(
                outputTensor.data_ptr(),
                recv_count,
                recv_type,
                root,
                comm,
                ccl::create_stream(stream.queue()));
          }

          return ret_evt;
        }
      },
      [](at::xpu::XPUStream&, c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {
      },
      [](at::xpu::XPUStream&, c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {
      },
      OpType::SCATTER);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce_impl(
    at::Tensor& tensor,
    const AllreduceOptions& opts) {
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ccl::allreduce_attr attr,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::event ret_evt;
        ret_evt = ccl::allreduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            ccl::create_stream(stream.queue()),
            attr);
        return ret_evt;
      },
      OpType::ALLREDUCE);
}

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
  check_xpu_single_tensor(tensor);
  TORCH_CHECK(
      !isFloat8Type(tensor.scalar_type()),
      "Float8 dtypes are not currenlty supported for XCCL reductions");

  return allreduce_impl(tensor, opts);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  check_xpu_tensors_same_device(tensors);
  TORCH_CHECK(
      !isFloat8Type(tensors.back().scalar_type()),
      "Float8 dtypes are not currenlty supported for XCCL reductions");

  return collectiveCoalesced(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ccl::allreduce_attr attr,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::event ret_evt;
        ret_evt = ccl::allreduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            ccl::create_stream(stream.queue()),
            attr);
        return ret_evt;
      },
      OpType::COALESCED);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    tensor = at::view_as_real(tensor);
  }
  check_xpu_single_tensor(tensor);

  const auto root = opts.rootRank + opts.rootTensor;

  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ccl::broadcast_attr attr,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ccl::event ret_evt;
        ret_evt = ccl::broadcast(
            input.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            root,
            comm,
            ccl::create_stream(stream.queue()),
            attr);
        return ret_evt;
      },
      OpType::BROADCAST);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::_broadcast_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const BroadcastOptions& opts) {
  if (outputTensor.numel() != inputTensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _broadcast_oop must have the same number of elements ");
  }
  const auto root = opts.rootRank + opts.rootTensor;
  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ccl::broadcast_attr attr,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ccl::event ret_evt;
        ret_evt = ccl::broadcast(
            input.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            root,
            comm,
            ccl::create_stream(stream.queue()),
            attr);
        return ret_evt;
      },
      OpType::BROADCAST);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::reduce(
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
  check_xpu_single_tensor(tensor);

  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ccl::reduce_attr attr,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        const int root = opts.rootRank + opts.rootTensor;
        const auto xcclDataType = getXcclDataType(input.scalar_type());
        const auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::event ret_evt;
        ret_evt = ccl::reduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            root,
            comm,
            ccl::create_stream(stream.queue()));
        return ret_evt;
      },
      OpType::REDUCE);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::_reduce_oop(
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
      [&](at::Tensor& input,
          at::Tensor& output,
          ccl::reduce_attr attr,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        const int root = opts.rootRank + opts.rootTensor;
        const auto xcclDataType = getXcclDataType(input.scalar_type());
        const auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::event ret_evt;
        ret_evt = ccl::reduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            root,
            comm,
            ccl::create_stream(stream.queue()));
        return ret_evt;
      },
      OpType::REDUCE);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto inputTensor = inputTensors.back();
  check_xpu_single_tensor(inputTensor);
  // @lint-ignore CLANGTIDY
  std::vector<at::Tensor>& outputTensors_ = outputTensors.back();

  bool same_size = check_same_size(outputTensors_);
  if (same_size) {
    // Flatten a vector of tensors into a single, stacked tensor.
    at::Tensor outputFlattened = newLikeFlat(outputTensors_);

    return collective(
        inputTensor,
        outputFlattened,
        [&](at::Tensor& input,
            at::Tensor& output,
            ccl::allgather_attr attr,
            xcclComm_t& comm,
            at::xpu::XPUStream& stream) {
          c10::xpu::XPUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          auto xcclDataType = getXcclDataType(input.scalar_type());
          ccl::event ret_evt;

          ret_evt = ccl::allgather(
              input.data_ptr(),
              output.data_ptr(),
              (size_t)input.numel(),
              xcclDataType,
              comm,
              ccl::create_stream(stream.queue()),
              attr);
          return ret_evt;
        },
        [](at::xpu::XPUStream&,
           c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {},
        [&](at::xpu::XPUStream& Stream,
            c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {
          // Copy the flattened output tensors to the outputs.
          c10::StreamGuard guard(Stream);
          for (const auto j : c10::irange(outputTensors_.size())) {
            c10::xpu::XPUCachingAllocator::recordStream(
                outputTensors_[j].storage().data_ptr(), Stream);
            outputTensors_[j].copy_(outputFlattened[j], true);
          }
        },
        OpType::ALLGATHER);
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

c10::intrusive_ptr<Work> ProcessGroupXCCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  check_xpu_single_tensor(input_tensor);
  check_xpu_single_tensor(output_tensor);

  if (input_tensor.dtype() != output_tensor.dtype()) {
    C10_THROW_ERROR(
        TypeError, "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "output tensor size must be equal to world_size times input tensor size");
  }

  return collective(
      input_tensor,
      output_tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ccl::allgather_attr attr,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        c10::xpu::XPUCachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ccl::event ret_evt;
        ret_evt = ccl::allgather(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            comm,
            ccl::create_stream(stream.queue()),
            attr);
        return ret_evt;
      },
      OpType::_ALLGATHER_BASE);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          ccl::allgather_attr attr,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ccl::event ret_evt;
        ret_evt = ccl::allgather(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            comm,
            ccl::create_stream(stream.queue()),
            attr);
        return ret_evt;
      },
      OpType::COALESCED);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto outputTensor = outputTensors.back();
  check_xpu_single_tensor(outputTensor);
  // @lint-ignore CLANGTIDY
  auto inputTensors_ = inputTensors.back();
  TORCH_CHECK(
      !isFloat8Type(outputTensor.scalar_type()),
      "Float8 dtypes are not currenlty supported for XCCL reductions");

  bool same_size = check_same_size(inputTensors_);
  if (same_size) {
    // Flatten a vector of tensors into a single, stacked tensor.
    at::Tensor inputFlattened = newLikeFlat(inputTensors_);
    return collective(
        inputFlattened,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            ccl::reduce_attr attr,
            xcclComm_t& comm,
            at::xpu::XPUStream& stream) {
          c10::xpu::XPUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          auto xcclDataType = getXcclDataType(input.scalar_type());
          auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
          ccl::event ret_evt;
          ret_evt = ccl::reduce_scatter(
              input.data_ptr(),
              output.data_ptr(),
              (size_t)output.numel(),
              xcclDataType,
              xcclReduceOp,
              comm,
              ccl::create_stream(stream.queue()));
          return ret_evt;
        },
        [&](at::xpu::XPUStream& Stream,
            c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {
          // Copy the input tensors to the flattened inputs.
          c10::StreamGuard guard(Stream);
          for (const auto j : c10::irange(inputTensors_.size())) {
            c10::xpu::XPUCachingAllocator::recordStream(
                inputTensors_[j].storage().data_ptr(), Stream);
            inputFlattened[j].copy_(inputTensors_[j], true);
          }
        },
        [&](at::xpu::XPUStream&,
            c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {},
        OpType::REDUCE_SCATTER);
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

c10::intrusive_ptr<Work> ProcessGroupXCCL::_reduce_scatter_base(
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
      "Float8 dtypes are not currenlty supported for XCCL reductions");

  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ccl::reduce_attr attr,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        c10::xpu::XPUCachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        auto xcclDataType = getXcclDataType(input.scalar_type());
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::event ret_evt;
        ret_evt = ccl::reduce_scatter(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)output.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            ccl::create_stream(stream.queue()));
        return ret_evt;
      },
      OpType::_REDUCE_SCATTER_BASE);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(
      !isFloat8Type(inputs.back().scalar_type()),
      "Float8 dtypes are not currenlty supported for XCCL reductions");
  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          ccl::reduce_attr attr,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        c10::xpu::XPUCachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        auto xcclDataType = getXcclDataType(input.scalar_type());
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::event ret_evt;
        ret_evt = ccl::reduce_scatter(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)output.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            ccl::create_stream(stream.queue()));
        return ret_evt;
      },
      OpType::COALESCED);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::barrier(const BarrierOptions& opts) {
  // Device to use for barrier
  int barDevIdx = -1;

  // See nccl barrier comments
  if (!opts.device_ids.empty()) {
    barDevIdx = opts.device_ids[0];
  } else if (getBoundDeviceId()) {
    barDevIdx = (*getBoundDeviceId()).index();
  } else if (!usedDeviceIdxs_.empty()) {
    barDevIdx = *usedDeviceIdxs_.begin();
  } else {
    barDevIdx =
        static_cast<int16_t>(rank_ % at::detail::getXPUHooks().getNumGPUs());
  }

  TORCH_CHECK_WITH(
      ValueError,
      barDevIdx >= 0,
      "Failed to infer a GPU device id to perform barrier. ");
  auto barDevice = at::Device(at::DeviceType::XPU, barDevIdx);

  at::Tensor barrierTensor =
      at::zeros({1}, at::TensorOptions().device(barDevice).dtype(at::kFloat));

  auto work = allreduce_impl(barrierTensor);

  auto xcclWork = dynamic_cast<ProcessGroupXCCL::WorkXCCL*>(work.get());
  TORCH_CHECK(xcclWork);
  xcclWork->barrierTensor_ = std::move(barrierTensor);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  check_xpu_single_tensor(outputTensor, true);
  check_xpu_single_tensor(inputTensor, true);
  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    TORCH_CHECK(
        outputTensor.numel() == inputTensor.numel() &&
            outputTensor.scalar_type() == inputTensor.scalar_type(),
        "xpu_alltoall_base: tensors are not equal in size or data type");
    TORCH_CHECK(
        outputTensor.size(0) % size_ == 0,
        "xpu_alltoall_base: tensor's dim 0 does not divide equally across group size");
    return collective(
        inputTensor,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            ccl::alltoall_attr attr,
            xcclComm_t& comm,
            at::xpu::XPUStream& stream) {
          c10::xpu::XPUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          auto xcclDataType = getXcclDataType(output.scalar_type());
          ccl::event ret_evt;
          ret_evt = ccl::alltoall(
              input.data_ptr(),
              output.data_ptr(),
              (size_t)output.numel() / comm.size(),
              xcclDataType,
              comm,
              ccl::create_stream(stream.queue()),
              attr);
          return ret_evt;
        },
        OpType::ALLTOALL_BASE);
  } else {
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);

    return collective(
        inputTensor,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            ccl::alltoallv_attr attr,
            xcclComm_t& comm,
            at::xpu::XPUStream& stream) {
          std::vector<size_t> sendCounts(size_);
          std::vector<size_t> recvCounts(size_);
          bool inputSplitsEqual = inputSplitSizes.size() == 0;
          bool outputSplitsEqual = outputSplitSizes.size() == 0;

          size_t inLen = input.numel();
          size_t outLen = output.numel();
          if (inLen)
            inLen /= (inputSplitsEqual ? size_ : input.size(0));
          if (outLen)
            outLen /= (outputSplitsEqual ? size_ : output.size(0));

          for (int i = 0; i < size_; i++) {
            sendCounts[i] =
                (inputSplitsEqual ? inLen : inputSplitSizes[i] * inLen);
            recvCounts[i] =
                (outputSplitsEqual ? outLen : outputSplitSizes[i] * outLen);
          }
          auto xcclDataType = getXcclDataType(output.scalar_type());
          ccl::event ret_evt;

          ret_evt = ccl::alltoallv(
              input.data_ptr(),
              sendCounts,
              output.data_ptr(),
              recvCounts,
              xcclDataType,
              comm,
              ccl::create_stream(stream.queue()),
              attr);
          return ret_evt;
        },
        OpType::ALLTOALL_BASE);
  }
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& /* unused */) {
  auto device = outputTensors[0].device();
  for (const auto r : c10::irange(outputTensors.size())) {
    check_xpu_single_tensor(outputTensors[r], true);
    check_xpu_single_tensor(inputTensors[r], true);
    TORCH_CHECK(
        device == outputTensors[r].device() &&
            device == inputTensors[r].device(),
        "Tensors must be on the same device")
  }

  return collective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ccl::alltoallv_attr attr,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        c10::OptionalStreamGuard stream_guard(stream.unwrap());
        at::Tensor flatInput;
        at::Tensor flatOutput;

        std::vector<size_t> sendCounts(size_);
        std::vector<size_t> recvCounts(size_);

        int64_t flatSendCount;
        int64_t flatRecvCount;

        bool isInputFlat = computeLengthsAndCheckAndGetFlat(
            inputTensors, sendCounts, flatInput, flatSendCount);
        bool isOutputFlat = computeLengthsAndCheckAndGetFlat(
            outputTensors, recvCounts, flatOutput, flatRecvCount);
        if (!isInputFlat) {
          auto flatInputSplits = flatInput.split_with_sizes(
              c10::IntArrayRef((int64_t*)sendCounts.data(), sendCounts.size()),
              0);

          for (int i = 0; i < size_; i++) {
            flatInputSplits[i].copy_(inputTensors[i].view({-1}));
          }
        }

        auto xcclDataType = getXcclDataType(flatOutput.scalar_type());
        ccl::event ret_evt;
        ret_evt = ccl::alltoallv(
            flatInput.data_ptr(),
            sendCounts,
            flatOutput.data_ptr(),
            recvCounts,
            xcclDataType,
            comm,
            ccl::create_stream(stream.queue()),
            attr);

        if (!isOutputFlat) {
          ret_evt.wait();
          auto flatOutputSplits = flatOutput.split_with_sizes(
              c10::IntArrayRef((int64_t*)recvCounts.data(), recvCounts.size()),
              0);

          for (int i = 0; i < size_; i++) {
            outputTensors[i].view({-1}).copy_(flatOutputSplits[i]);
          }
        }

        stream.synchronize();
        return ret_evt;
      },
      [](at::xpu::XPUStream&, c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {
      },
      [](at::xpu::XPUStream&, c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {
      },
      OpType::ALLTOALL);
}

} // namespace c10d

#endif // USE_C10D_XCCL
