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
#include <torch/csrc/distributed/c10d/logger.hpp>
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

void check_xpu_single_tensor(const at::Tensor& tensor) {
  if (!tensor.is_xpu() || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be XPU and dense");
  }
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
  }
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
// Get a key string from device
inline std::string getKeyFromDevice(at::Device& device) {
  return std::to_string(device.index());
}

inline at::DeviceIndex getIndexFromDeviceKey(const std::string& deviceKey) {
  // initialize the device index to -1, which is an invalid value.
  int index = -1;
  try {
    index = std::stoi(deviceKey);
  } catch (const std::invalid_argument& e) {
    LOG(ERROR) << c10::str(
        "Invalid deviceKey: ", deviceKey, ",", e.what(), ".");
  } catch (const std::out_of_range& e) {
    LOG(ERROR) << "Out of range: " << e.what();
  }
  return static_cast<at::DeviceIndex>(index);
}

std::string getKeySendRecv(int myRank, int peer) {
  int lowRank = myRank < peer ? myRank : peer;
  int highRank = myRank < peer ? peer : myRank;
  std::string sendRecvPair =
      std::to_string(lowRank) + ":" + std::to_string(highRank);
  return sendRecvPair;
}

// Get device from tensor
inline at::Device getDevice(at::Tensor& tensor) {
  return tensor.device();
}

void syncStream(
    at::Device& device,
    c10::Event& ncclEvent,
    c10::Stream& ncclStream) {
  ncclEvent.record(at::xpu::getCurrentXPUStream(device.index()));
  ncclEvent.block(ncclStream);
}

} // namespace

static std::mutex xcclCommDevIdxMapMutex;
static std::unordered_map<std::shared_ptr<xcclComm_t>, int> xcclCommDevIdxMap;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;
thread_local uint64_t ProcessGroupXCCL::ncclActiveGroupCounter_ = 0;

std::ostream& operator<<(
    std::ostream& output,
    const ProcessGroupXCCL::WorkXCCL& WorkXCCL) {
  std::string workInfo;
  return output << workInfo;
}

ProcessGroupXCCL::WorkXCCL::WorkXCCL(
    const std::string& pgUID,
    const std::string& pgDesc,
    at::Device& device,
    int rank,
    OpType opType,
    uint64_t seq,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputs,
    bool enableTiming,
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
//    ncclStartEvent_ = enableTiming
//        ? ProcessGroupXCCL::CUDAEventCache::get().create(enableTiming)
//        : nullptr;
//    ncclEndEvent_ =
//        ProcessGroupXCCL::CUDAEventCache::get().create(enableTiming);
//  } else {
//    ncclStartEvent_ = enableTiming
//        ? std::make_shared<at::cuda::CUDAEvent>(cudaEventDefault)
//        : nullptr;
//    ncclEndEvent_ = std::make_shared<at::cuda::CUDAEvent>(
//        enableTiming ? cudaEventDefault : cudaEventDisableTiming);
//  }
  unsigned char enable_timing = 0;
  ncclStartEvent_ = std::make_shared<at::xpu::XPUEvent>(enable_timing);
  ncclEndEvent_ = std::make_shared<at::xpu::XPUEvent>(enable_timing);
}

ProcessGroupXCCL::WorkXCCL::WorkXCCL(const WorkXCCL& w)
    : Work(w.rank_, w.opType_),
      std::enable_shared_from_this<WorkXCCL>(w),
      pgUID_(w.pgUID_),
      pgDesc_(w.pgDesc_),
      device_(w.device_),
      ncclStartEvent_(w.ncclStartEvent_),
      ncclEndEvent_(w.ncclEndEvent_),
      cclComm_(w.cclComm_),
      blockingWait_(w.blockingWait_),
      opTimeout_(w.opTimeout_),
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

ProcessGroupXCCL::WorkXCCL::~WorkXCCL() = default;

bool ProcessGroupXCCL::WorkXCCL::isCompleted() {
    return true;
}

bool ProcessGroupXCCL::WorkXCCL::isSuccess() const {
  C10_THROW_ERROR(NotImplementedError, "Work::isSuccess() is deprecated");
}

const std::string& ProcessGroupXCCL::WorkXCCL::logPrefix() const {
  static std::string prefix = c10::str("[Rank ", rank_, "] ");
  return prefix;
}

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

bool ProcessGroupXCCL::WorkXCCL::isCompleted() {
  if (xcclEndEvent_ && xcclEndEvent_->query()) {
    return true;
  }
  return false;
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
  return true;
}

void ProcessGroupXCCL::WorkXCCL::abort() {
  // Abort all communicators of this work
//  ncclCommAbort(cclComm_);

  ncclCommDevIdxMapMutex.lock();
  ncclCommDevIdxMap.erase(cclComm_);
  ncclCommDevIdxMapMutex.unlock();
}

static std::atomic<size_t> process_group_id = 0;

constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple. You are probably using multiple "
    "devices under one thread. The support for such usage has been deprecated. "
    "For details, please refer to "
    "https://pytorch.org/docs/stable/distributed.html#multi-gpu-collective-functions. "
    "ProcessGroupXCCL continues supporting multi-process and multi-thread modes.";

ProcessGroupXCCL::ProcessGroupXCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(options),
      ncclCommCounter_(0),
      terminateProcessGroup_(false),
      collectiveDebugInfoMode_(false),
      local_id_(process_group_id++) {
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

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupXCCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType,
    const char* profilingTitle,
    bool avoidRecordStreams,
    bool nanCheck) {
  using traits = function_traits<Fn>;
  using attr_t = typename traits::template arg<2>::type;
  attr_t attr = ccl::create_operation_attr<attr_t>();

  auto device = input.device();
  const auto key = std::to_string(device.index());
  auto comm = getXCCLComm(key, device);

  auto stream = xcclStreams_.at(key);
  std::vector<at::Tensor> outputs{output};

  c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;

  work = initWork(device, rank_, opType);

  work->outputs_ =
      std::make_shared<std::vector<at::Tensor>>(std::move(outputs));
  c10::xpu::XPUCachingAllocator::recordStream(
      input.storage().data_ptr(), stream);

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

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1, "Expecting one tensor only but got multiple");
  auto tensor = tensors.back();
  check_xpu_single_tensor(tensor);
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ccl::allreduce_attr attr,
          xcclComm_t& comm,
          ccl::stream& stream) {
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
        return ret_evt;
      },
      OpType::ALLREDUCE);
}

} // namespace c10d

#endif // USE_C10D_XCCL
