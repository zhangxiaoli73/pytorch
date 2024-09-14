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
#include <variant>

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
#include <torch/torch.h>

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

XCCL_KVS kvs;
std::mutex kvs_mutex;

XCCL_KVS get_kvs(int rank, c10d::Store& store) {
  std::lock_guard<std::mutex> lock(kvs_mutex);
  if (kvs)
    return kvs;
  std::string storeKey = "ccl_kvs";

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

bool check_same_size(const std::vector<at::Tensor>& input_tensors) {
  for (const auto& input_tensor : input_tensors) {
    if (!input_tensors[0].is_same_size(input_tensor)) {
      return false;
    }
  }
  return true;
}

void check_xpu_single_tensor(const at::Tensor& tensor) {
  if (!tensor.is_xpu() || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be XPU and dense");
  }
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
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

} // namespace

static std::mutex xcclCommDevIdxMapMutex;
static std::unordered_map<std::shared_ptr<xcclComm_t>, int> xcclCommDevIdxMap;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;

// Before implementing send/recv, the xcclActiveGroupCounter_ variable has no
// effect.
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

bool ProcessGroupXCCL::WorkXCCL::checkTimeout(
    std::optional<std::chrono::milliseconds> timeout) {
  auto currentTimepoint = std::chrono::steady_clock::now();
  auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTimepoint - workStartTime_);
  std::chrono::milliseconds opTimeout = std::chrono::milliseconds(60000);

  auto workTimeout = timeout ? *timeout : opTimeout;

  if (timeElapsed < workTimeout)
    return false;
  return true;
}

void ProcessGroupXCCL::WorkXCCL::finishWorkXcclError(
    const std::exception_ptr& eptr) {
  future_->setError(eptr);
  finish(eptr);
}

bool ProcessGroupXCCL::WorkXCCL::isCompleted() {
  for (auto& ret : rets) {
    bool flag;
    try {
      TORCH_CHECK(flag = ret.test());
    } catch (...) {
      finishWorkXcclError(std::current_exception());
      return true;
    }
    if (!flag) {
      return false;
    }
  }
  return true;
}

void ProcessGroupXCCL::WorkXCCL::synchronize() {
  synchronizeInternal(kNoTimeout);
}

void ProcessGroupXCCL::WorkXCCL::synchronizeStream() {
  auto currentStream = at::xpu::getCurrentXPUStream(device_.index());
  // Block the current stream on the XCCL stream
  xcclEndEvent_->block(currentStream);
}

void ProcessGroupXCCL::WorkXCCL::synchronizeInternal(
    std::chrono::milliseconds timeout) {
  synchronizeStream();

  if (blockingWait_) {
    while (!isCompleted()) {
      bool timedOut = checkTimeout(
          timeout == kNoTimeout ? std::nullopt : std::make_optional(timeout));
      if (timedOut) {
        break;
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
  }
}

bool ProcessGroupXCCL::WorkXCCL::wait(std::chrono::milliseconds timeout) {
  synchronizeInternal(timeout);
  for (auto& event : rets) {
    event.wait();
  }
  rets.clear();
  return true;
}

constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple. You are probably using multiple "
    "devices under one thread. The support for such usage has been deprecated. "
    "For details, please refer to "
    "https://pytorch.org/docs/stable/distributed.html#multi-gpu-collective-functions. "
    "ProcessGroupXCCL continues supporting multi-process and multi-thread modes.";

ProcessGroupXCCL::ProcessGroupXCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : Backend(rank, size), store_(store) {
  blockingWait_ = getCvarBool(TORCH_XCCL_BLOCKING_WAIT, false);
  init();

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
    at::Device& device) {
  if (deviceKey.empty()) {
    C10_THROW_ERROR(
        DistBackendError,
        "Not able to create/get the XCCL Communicator since "
        "the devices are empty ");
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (devXCCLCommMap_.find(deviceKey) != devXCCLCommMap_.end()) {
      return devXCCLCommMap_[deviceKey];
    }
  }

  std::shared_ptr<xcclComm_t> XCCLComm;

  XCCL_KVS kvs = get_kvs(rank_, *store_);

  int numRanks, rank;
  numRanks = getSize();
  rank = getRank();

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

  xcclStreams_.emplace(deviceKey, std::move(stream));

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
static constexpr int CoalActive = 0x01, CoalColl = 0x02;
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

// align with single-device style, input_t and output_t due to
// allgatherv need vector output
template <
    typename Fn,
    typename input_t,
    typename output_t,
    typename PreProcess,
    typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupXCCL::collective(
    std::vector<input_t>& inputs,
    std::vector<output_t>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType) {
  using traits = function_traits<Fn>;
  using attr_t = typename traits::template arg<2>::type;
  attr_t attr = ccl::create_operation_attr<attr_t>();

  auto device = inputs[0].device();
  const auto key = std::to_string(device.index());
  auto comm = getXCCLComm(key, device);

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

  c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;

  work = initWork(device, rank_, opType);

  { // Do we need to store the result of the operation?
    std::variant<std::vector<at::Tensor>, std::vector<std::vector<at::Tensor>>>
        outputs;
    std::visit(
        [&work](auto&& outputData) {
          using T = std::decay_t<decltype(outputData)>;

          if constexpr (std::is_same_v<T, std::vector<at::Tensor>>) {
            work->outputs_ = std::make_shared<std::vector<at::Tensor>>(
                std::move(outputData));
          } else if constexpr (std::is_same_v<
                                   T,
                                   std::vector<std::vector<at::Tensor>>>) {
            std::vector<at::Tensor> flattened;
            for (auto& vec : outputData) {
              flattened.insert(flattened.end(), vec.begin(), vec.end());
            }
            work->outputs_ =
                std::make_shared<std::vector<at::Tensor>>(std::move(flattened));
          }
        },
        outputs);
  }

  pre(stream, work);

  for (const auto& input : inputs) {
    c10::xpu::XPUCachingAllocator::recordStream(
        input.storage().data_ptr(), stream);
  }

  work->addResult(fn(inputs[0], outputs[0], attr, *comm, stream));

  post(stream, work);

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

template <
    typename Fn,
    typename input_t,
    typename output_t,
    typename PreProcess,
    typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupXCCL::collective(
    input_t& input,
    output_t& output,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType) {
  auto inputs = std::vector<input_t>{input};
  auto outputs = std::vector<output_t>{output};
  return collective(inputs, outputs, fn, pre, post, opType);
}

template <typename Fn, typename input_t, typename output_t>
c10::intrusive_ptr<Work> ProcessGroupXCCL::collective(
    input_t& input,
    output_t& output,
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
  auto comm = getXCCLComm(key, device);

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

  c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;

  work = initWork(device, rank_, opType);

  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  {
    AutoXcclGroup xccl_group_guard;
    for (const auto i : c10::irange(inputs.size())) {
      c10::xpu::XPUCachingAllocator::recordStream(
          inputs[i].storage().data_ptr(), stream);
      work->addResult(fn(inputs[i], outputs[i], attr, *comm, stream));
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

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  check_xpu_single_tensor(tensor);
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
    // xccl implemented allgatherv, so broadcast_oop not needed
    return collective(
        inputTensor,
        outputTensors_,
        [=](at::Tensor& input,
            const std::vector<at::Tensor>& outputs,
            ccl::allgatherv_attr attr,
            xcclComm_t& comm,
            at::xpu::XPUStream& stream) {
          ccl::event ret_evt;
          auto xcclDataType = getXcclDataType(input.scalar_type());

          std::vector<size_t> recvCounts(outputs.size(), 0);
          std::transform(
              outputs.begin(),
              outputs.end(),
              recvCounts.begin(),
              [](const at::Tensor& t) { return t.numel(); });

          TORCH_CHECK(
              (size_t)input.numel() == recvCounts[rank_],
              "allgather: send and recv count doesn't match");

          std::vector<void*> recvBufs(outputs.size(), nullptr);
          std::transform(
              outputs.begin(),
              outputs.end(),
              recvBufs.begin(),
              [](const at::Tensor& t) { return t.data_ptr(); });

          ret_evt = ccl::allgatherv(
              input.data_ptr(),
              (size_t)input.numel(),
              recvBufs,
              recvCounts,
              xcclDataType,
              comm,
              ccl::create_stream(stream.queue()),
              attr);
          return ret_evt;
        },
        c10d::OpType::ALLGATHER);
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

} // namespace c10d

#endif // USE_C10D_XCCL
