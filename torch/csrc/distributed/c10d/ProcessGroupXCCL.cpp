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
    C10_THROW_ERROR(ValueError, "Unhandled ReduceOp");
  }
}

} // namespace

static std::mutex xcclCommDevIdxMapMutex;
static std::unordered_map<std::shared_ptr<xcclComm_t>, int> xcclCommDevIdxMap;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;

ProcessGroupXCCL::WorkXCCL::WorkXCCL(
    at::Device& device,
    int rank,
    OpType opType,
    const std::optional<std::vector<at::Tensor>>& inputs)
    : WorkGCCL(rank, opType, "profilingTitle", inputs),
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

ProcessGroupXCCL::ProcessGroupXCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : ProcessGroupGCCL(rank, size), store_(store) {
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
  numRanks = this->getSize();
  rank = this->getRank();

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


ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::allreduceImpl(at::Tensor& input,
    at::Tensor& output, ncclComm_t comm, c10::Stream& stream) {
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
        stream,
        attr);
    return ncclSuccess;
}

ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::broadcastImpl(at::Tensor& input,
    at::Tensor& output, ncclComm_t comm, c10::Stream& stream) {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}
ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::reduceImpl(at::Tensor& input,
    at::Tensor& output, ncclComm_t comm, c10::Stream& stream) {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}
ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::allgatherImpl(at::Tensor& input,
    at::Tensor& output, ncclComm_t comm, c10::Stream& stream) {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}
ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::reducescatterImpl(at::Tensor& input,
    at::Tensor& output, ncclComm_t comm, c10::Stream& stream) {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}
ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::all2allImpl(at::Tensor& input,
    at::Tensor& output, ncclComm_t comm, c10::Stream& stream) {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}
ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::sendImpl(at::Tensor& input,
    ncclComm_t comm, c10::Stream& stream, int dst) {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}
ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::recvImpl(at::Tensor& output,
    ncclComm_t comm, c10::Stream& stream, int dst) {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}
ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::gatherImpl(at::Tensor& output,
    ncclComm_t comm, c10::Stream& stream, int dst) {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}
ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::scatterImpl(at::Tensor& input,
    at::Tensor& output, ncclComm_t comm, c10::Stream& stream) {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}
ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::groupStartImpl(std::shared_ptr<NCCLComm> comm) {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}
ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::groupEndImpl() {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}
ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::dataTypes {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}
ncclResult_t ProcessGroupXCCL::CollectivesXCCLImpl::reductionTypes {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}

template <typename Fn>
c10::intrusive_ptr<Work> collectiveCoalesced(
    std::vector<at::Tensor>& input,
    std::vector<at::Tensor>& output,
    Fn fn,
    OpType opType,
    const char* profilingTitle = nullptr,
    bool avoidRecordStreams = false) {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> pointToPoint(
    at::Tensor& tensor,
    Fn fn,
    int peer,
    OpType opType,
    PreProcess pre,
    PostProcess post,
    const char* profilingTitle = nullptr) {
    TORCH_CHECK(false, "ProcessGroupXCCL allreduce not implemented");
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
      bool nanCheck = true) {
  using traits = function_traits<Fn>;
  using attr_t = typename traits::template arg<2>::type;
  attr_t attr = ccl::create_operation_attr<attr_t>();

  auto device = inputs[0].device();
  const auto key = std::to_string(device.index());
  auto comm = getXCCLComm(key, device);

  auto stream = xcclStreams_.at(key);

  c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;

  work = initWork(device, rank_, opType);

  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(std::move(outputs));
  c10::xpu::XPUCachingAllocator::recordStream(input.storage().data_ptr(), stream);

  auto ccl_stream = ccl::create_stream(stream.queue());

  fn(input, output, attr, *comm, ccl_stream);

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

} // namespace c10d

#endif // USE_C10D_XCCL
