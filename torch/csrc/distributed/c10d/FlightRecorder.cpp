


const int64_t FlightRecoder::kWatchdogThreadSleepMillis = 100;

FlightRecoder::FlightRecoder() {


}
void FlightRecoder::watchdogHandler() {
  bool done = false;
  lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
  auto lastStatusUpdateTime = std::chrono::steady_clock::now();
  std::list<c10::Work> completedWorkList;

  while (!done || !terminateProcessGroup_.load()) {
    std::unique_lock<std::mutex> lock(workMetaListMutex_);
    // We busy-poll the work vector every kWatchdogThreadSleepMillis
    // milliseconds as long as the atomic is True.
    workMetaListCV_.wait_for(
        lock,
        std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool { return terminateProcessGroup_.load(); });
    // Bump up heart beat by one.
    heartbeat_++;

// Some versions of GLOG support less-spammy version of LOG_EVERY_MS
// in which case we don't want to spam the logs.
#ifdef LOG_EVERY_MS
    // Log the progress of this PG periodically
    C10_LOG_EVERY_MS(INFO, kWorkStatusUpdatePeriodMs) << c10::str(
        logPrefix(),
        "NCCL Work update periodically: ",
        "last enqueued NCCL work: ",
        pgStatus_->lastEnqueuedSeq,
        ", last completed NCCL work: ",
        pgStatus_->lastCompletedSeq,
        ".");
#endif
    auto logger = ::c10d::C10dLogger::getLogger();
    if (logger &&
        computeDeltaMS(
            lastStatusUpdateTime, std::chrono::steady_clock::now()) >=
            kWorkStatusUpdatePeriodMs) {
      ::c10d::C10dLoggingData data;
      // logging integers
      data.integers["pg_id"] = pgStatus_->pg_id;
      data.integers["rank"] = pgStatus_->rank;
      data.integers["global_rank"] = pgStatus_->global_rank;
      data.integers["last_enqueued_work"] = pgStatus_->lastEnqueuedSeq;
      data.integers["last_started_work"] = pgStatus_->lastStartedSeq;
      data.integers["last_completed_work"] = pgStatus_->lastCompletedSeq;
      data.integers["last_enqueued_numel_in"] = pgStatus_->lastEnqueuedNumelIn;
      data.integers["last_enqueued_numel_out"] =
          pgStatus_->lastEnqueuedNumelOut;
      data.integers["last_completed_numel_in"] =
          pgStatus_->lastCompletedNumelIn;
      data.integers["last_completed_numel_out"] =
          pgStatus_->lastCompletedNumelOut;
      // logging strings
      data.strings["last_enqueued_work_name"] = pgStatus_->lastEnqueuedWorkName;
      data.strings["last_started_work_name"] = pgStatus_->lastStartedWorkName;
      data.strings["last_completed_work_name"] =
          pgStatus_->lastCompletedWorkName;
      data.strings["pg_name"] = pgStatus_->pg_uid_;
      data.strings["pg_desc"] = pgStatus_->pg_desc_;
      logger->log(data);
      lastStatusUpdateTime = std::chrono::steady_clock::now();
    }

    // record some information
    for (auto it = workMetaList_.begin(); it != workMetaList_.end();
     /* no increment */) {
         auto& work = *it;
         recoder
     }


    for (auto it = workMetaList_.begin(); it != workMetaList_.end();
         /* no increment */) {
      auto& work = *it;
      // When terminateProcessGroup_ is true, communicators have already been
      // aborted, So cannot check exception based on them. But watchdog needs to
      // finish the check for the works that have already been enqueued to
      // workMetaList_
      if (!terminateProcessGroup_.load()) {
        work.checkAndSetException();
      }
      bool timedOut = work.checkTimeout();

      // If work hits an exception (either an error or timeout)
      if (work.exception()) {
        // log as soon as exception is detected
        LOG(ERROR) << c10::str(
            logPrefix(),
            "Exception (either an error or timeout) detected by watchdog at work: ",
            work.seq_,
            ", last enqueued NCCL work: ",
            pgStatus_->lastEnqueuedSeq,
            ", last completed NCCL work: ",
            pgStatus_->lastCompletedSeq,
            ".");
        // try to notify other ranks via global TCPStore to dump the flight
        // recorder when a collective timeout or exception happens. Flight
        // recorder behavior is independent of desync Debug.
        if (dumpOnTimeoutOrEx_) {
          try {
            auto rank = pgStatus_->global_rank;
            auto vec = std::vector<uint8_t>(
                reinterpret_cast<uint8_t*>(&rank),
                reinterpret_cast<uint8_t*>(&rank) + sizeof(rank));
            // todo: zl_debug not needed in some backend???
            globalStore_->set(std::string(EXCEPTION_DUMP), vec);
            if (!shouldDump_.load()) {
              LOG(ERROR)
                  << logPrefix()
                  << "Broadcasting flight-recorder dump signal to other processes via TCPStore.";
            }
            // signal the monitor thread on PG0 to start dumping
            shouldDump_.store(true);
            if (sleepAfterException_) {
              // This sleep is used to give time for dumping before throwing
              // exception
              std::this_thread::sleep_for(
                  std::chrono::seconds(heartbeatTimeoutInSec_));
              LOG(INFO) << logPrefix() << "slept for " << heartbeatTimeoutInSec_
                        << " giving time for flight recorder dumps to finish.";
            }
          } catch (const std::exception& e) {
            LOG(ERROR) << logPrefix()
                       << "Failed to set dump signal in tcpstore. "
                       << "Error: " << e.what();
          }
        }

        if (SHOULD_CLEAN_UP(asyncErrorHandling_)) {
          // Abort work and corresponding communicators
          work.abort();
          // PG level abort, which would abort all other communicators on this
          // rank
          abort();
        }

        // Report desync state in case of timeout
        if (timedOut) {
          LOG(ERROR) << c10::str(
              logPrefix(),
              "Timeout at NCCL work: ",
              work.seq_,
              ", last enqueued NCCL work: ",
              pgStatus_->lastEnqueuedSeq,
              ", last completed NCCL work: ",
              pgStatus_->lastCompletedSeq,
              ".");
          if (desyncDebug_) {
            try {
              collectiveDebugInfoMode_.store(true);
              auto desyncMsg = getNCCLWatchdogDebugInfo();
              LOG(ERROR) << logPrefix() << desyncMsg;
            } catch (const std::exception& e) {
              LOG(ERROR)
                  << logPrefix()
                  << "Failed to retrieve TORCH_NCCL_DESYNC_DEBUG report. "
                  << " Please file an issue. Error: " << e.what();
            } catch (...) {
              LOG(ERROR)
                  << logPrefix()
                  << "Failed to rerieve TORCH_NCCL_DESYNC_DEBUG report with unknown error."
                  << " Please file an issue.";
            }
          }
        }
        // Throw exception
        work.handleException(asyncErrorHandling_);
      }

      // Work status logging for desync debug
      if (desyncDebug_) {
        if (work.isStarted()) {
          logWorkStart(work);
        }
        if (work.isCompleted()) {
          logWorkEnd(work);
        }
      }

      // a work could be started but not completed, so we should not update
      // lastStartedSeq and lastStartedOpName if the work state is checked
      // multiple times after the start
      if (pgStatus_->lastStartedSeq < static_cast<int64_t>(work.seq_) &&
          work.isStarted()) {
        pgStatus_->lastStartedSeq = work.seq_;
        pgStatus_->lastStartedWorkName = opTypeToString(work.opType_);
      }

      // Clean up completed work
      if (work.isCompleted()) {
        {
          // Reset the timeout and first work if the work is completed.
          std::lock_guard<std::mutex> timeoutLock(mtxTimeoutExtension_);
          if (work.ownedEphermeralTimeout_.count() > 0) {
            ephemeralTimeoutActive_ -= work.ownedEphermeralTimeout_;
            ephemeralTimeoutInflight_ -= work.ownedEphermeralTimeout_;
          }
        }
        pgStatus_->lastCompletedSeq = work.seq_;
        pgStatus_->lastCompletedWorkName = opTypeToString(work.opType_);
        pgStatus_->lastCompletedNumelIn = work.numelIn_;
        pgStatus_->lastCompletedNumelOut = work.numelOut_;
        NCCLTraceBuffer::get()->retire_id(work.trace_id_, true);
        if (onCompletionHook_) {
          // Move Work object to completedWorkList_ to be consumed by the hook
          // thread
          {
            const std::lock_guard<std::mutex> lock(completedWorkListMutex_);
            completedWorkList_.splice(
                completedWorkList_.end(), workMetaList_, it++);
          }
          completedWorkListCV_.notify_one();
        } else {
          it = workMetaList_.erase(it);
          lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
        }
        at::cuda::CUDAGraph::dec_pending_event_queries();
      } else {
        // Increment the iterator if the current WorkNCCL object is not
        // completed.
        ++it;
      }
      // Increment heartbeat after each work processed,
      // in case processing is slowed down (but not hung) by cuda api contention
      heartbeat_++;
    }
    done = workMetaList_.empty();
  }
}

std::string dump_nccl_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  return NCCLTraceBuffer::get()->dump(
      ncclDumpMap, includeCollectives, includeStackTraces, onlyActive);
}

std::string dump_nccl_trace_json(bool includeCollectives, bool onlyActive) {
  return NCCLTraceBuffer::get()->dump_json(
      ncclDumpMap, includeCollectives, onlyActive);
}

std::optional<std::function<void(std::function<void(const std::string&)>)>>&
get_cpp_trace_dumper() {
  static std::optional<
      std::function<void(std::function<void(const std::string&)>)>>
      dumper(std::nullopt);
  return dumper;
}

gil_checker_t& get_gil_checker() {
  static gil_checker_t gil_checker = nullptr;
  return gil_checker;
}

std::future<bool> launchAsyncGilCheck() {
  std::promise<bool> resultPromise;
  std::future<bool> resultFuture = resultPromise.get_future();
  TORCH_CHECK(get_gil_checker(), "Can't check GIL with null GIL checker");
  std::thread workerThread([promise = std::move(resultPromise)]() mutable {
    c10::setThreadName("pt_nccl_gil_chk");

    try {
      auto& gil_checker = get_gil_checker();
      promise.set_value((*gil_checker)());
    } catch (...) {
      promise.set_exception(std::current_exception());
    }
  });

  // Detach the thread to allow it to run independently
  workerThread.detach();

  return resultFuture;
}

std::string FlightRecoder::getNCCLWatchdogTimeoutErrorMsg(
    const std::string& extraMsg) {
  return c10::str(
      logPrefix(),
      "Received a dump signal due to a collective timeout from ",
      extraMsg,
      " and we will try our best to dump the debug info. ",
      "Last enqueued NCCL work: ",
      pgStatus_->lastEnqueuedSeq,
      ", last completed NCCL work: ",
      pgStatus_->lastCompletedSeq,
      ".",
      "This is most likely caused by incorrect usages of collectives, e.g., wrong ",
      "sizes used across ranks, the order of collectives is not same for all ranks ",
      "or the scheduled collective, for some reason, didn't run. Additionally, ",
      "this can be caused by GIL deadlock or other reasons such as network errors or ",
      "bugs in the communications library (e.g. NCCL), etc. ");
}

std::string FlightRecoder::getNCCLWatchdogTimeoutExitMsg(
    const std::string& exitReason) {
  return c10::str(
      logPrefix(),
      "Terminating the process after attempting to dump debug info, due to ",
      exitReason,
      ".");
}

void FlightRecoder::heartbeatMonitor() {
  c10::setThreadName("pt_nccl_heartbt");

  uint64_t heartBeatCounter = 0ULL;
  std::string errorMsg;
  std::string exitReason;
  bool checkDumpSignal = (dumpOnTimeoutOrEx_ && local_id_ == 0);
  int monitorPollInterval = checkDumpSignal ? coordCheckIntervalMilSec_
                                            : heartbeatTimeoutInSec_ * 1000;
  auto lastTimePollStore = std::chrono::steady_clock::now();
  auto lastTimeHeartBeatCheck = std::chrono::steady_clock::now();
  std::optional<DumpPipe> dumpPipe = std::nullopt;
  if (local_id_ == 0) {
    // DumpPipe is one per-trainer process, and its convenient to name them
    // after 'global' ranks in the system, So we assume processgroup (uid)==0 is
    // the global PG and has globally unique rank ids across trainers.
    dumpPipe.emplace(rank_);
  }
  while (true) {
    // This won't have any lock since this lock is only used here.
    // Please be aware that mutex `monitorMutex_` should not be used
    // somewhere else to avoid the deadlock.
    std::unique_lock<std::mutex> lock(monitorMutex_);
    if (monitorWakeUpCV_.wait_for(
            lock, std::chrono::milliseconds(monitorPollInterval), [&] {
              return terminateHeartbeatMonitorThread_.load();
            })) {
      // For the normal complete or user interception, monitorWakeUpCV_
      // will get notified, we early return and exit heartbeatMonitor.
      return;
    }
    auto currentTime = std::chrono::steady_clock::now();

    // We put extra functionality in the thread for the default PG (aka,
    // local_id_=0) because the signal is same across different PGs. We only
    // need to run once per process to avoid duplicate things performed in too
    // many separate threads. For example, we check a global flag on the
    // TCPStore periodically to see if any PG on any rank observed a timeout and
    // signaled peers to dump debugging info, and we avoid hammering the
    // TCPStore from all PGs on the same rank.
    if (checkDumpSignal) {
      // There are two scenarios where monitor thread will dump on timeout:
      // 1. The current rank is the first to observe a timeout in watchdog.
      // (shouldDump_ was set to true by the watchdog thread).
      // 2. Other ranks detected the timeout and signal the current rank to
      // dump. In addtion, monitor threads will dump if watchdog threads has no
      // heartbeat or dumpPipe is not empty.
      if (shouldDump_.load()) {
        errorMsg = getNCCLWatchdogTimeoutErrorMsg("this local rank");
        exitReason = "collective timeout or exception";
        break;
      }
      // We poll store to see if some ranks have flagged a timeout when
      // we haven't polled for `heartbeat_timeout` seconds and there haven't
      // any work added or removed for `watchdog_timeout` seconds.
      if (computeDeltaMS(lastWorkListUpdateTime_, currentTime) >=
              kWatchdogThreadSleepMillis &&
          computeDeltaMS(lastTimePollStore, currentTime) >=
              coordCheckIntervalMilSec_) {
        lastTimePollStore = currentTime;
        // Wrap globalStore_->check() in a try-catch block to avoid crashing if
        // the store is not available.
        bool checkExceptionDump = false;
        try {
          checkExceptionDump =
              globalStore_->check({std::string(EXCEPTION_DUMP)});
        } catch (const std::exception& e) {
          LOG(WARNING)
              << logPrefix()
              << "Failed to check the \"should dump\" flag on TCPStore, "
              << "(maybe TCPStore server has shut down too early), with error: "
              << e.what();
          // We give up for now assuming TCPStore has been torn down.
          return;
        }

        if (checkExceptionDump) {
          int timeOutRank = -1;
          if (!shouldDump_.load()) {
            LOG(ERROR)
                << logPrefix()
                << "Observed flight recorder dump signal from another rank via TCPStore.";
          }
          shouldDump_.store(true);
          try {
            auto vec = globalStore_->get(std::string(EXCEPTION_DUMP));
            TORCH_CHECK_WITH(
                DistBackendError,
                vec.size() == sizeof(int),
                "Invalid size for the timeout rank ID");
            std::memcpy(&timeOutRank, vec.data(), vec.size());
          } catch (const std::exception& e) {
            LOG(ERROR) << logPrefix()
                       << "Failed to get timeout rank ID from TCPStore."
                       << e.what();
          }
          errorMsg =
              getNCCLWatchdogTimeoutErrorMsg(c10::str(" rank ", timeOutRank));
          exitReason = "collective timeout or exception";
          break;
        }
      }
    }

    if (computeDeltaMS(lastTimeHeartBeatCheck, currentTime) >=
        heartbeatTimeoutInSec_ * 1000) {
      // Check the heart beat of watchdog thread.
      lastTimeHeartBeatCheck = currentTime;
      auto heartbeat = heartbeat_.load();
      if (heartbeat != heartBeatCounter) {
        heartBeatCounter = heartbeat;
      } else {
        shouldDump_.store(true);
        // Watchdog heartbeat timeout.
        errorMsg = c10::str(
            logPrefix(),
            "ProcessGroupNCCL's watchdog got stuck for ",
            heartbeatTimeoutInSec_,
            " seconds without making progress in monitoring enqueued collectives. ",
            "This typically indicates a NCCL/CUDA API (e.g., CudaEventDestroy) hang blocking the watchdog, ",
            "and could be triggered by another thread holding the GIL inside a ",
            "CUDA api (for example, CudaEventDestroy), or other deadlock-prone behaviors.",
            "If you suspect the watchdog is not actually stuck and a longer timeout would help, ",
            "you can either increase the timeout (TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC) to a larger value "
            "or disable the heartbeat monitor (TORCH_NCCL_ENABLE_MONITORING=0)."
            "If either of aforementioned helps, feel free to file an issue to PyTorch about the short timeout "
            "or false positive abort; otherwise, please attempt to debug the hang. ");
        exitReason = "ProcessGroupNCCL watchdog hang";
        break;
      }
    }
    // process a request to dump the trace. only PG uid 0 will respond to dump
    // requests, but this is fine since all PG's feed into the same flight
    // recorder and dump. After dump, the training should continue.
    if (dumpPipe.has_value() && dumpPipe->shouldDump()) {
      // best effort dump, not waiting for the dump here
      std::future<bool> fut = std::async(
          std::launch::async, [this]() { return this->dumpDebuggingInfo(); });
    }
  }
  LOG(ERROR) << errorMsg;

  auto& cpp_dumper = get_cpp_trace_dumper();
  if (logCppStackOnUncleanShutdown_ && cpp_dumper.has_value()) {
    LOG(INFO) << "Dumping c++ stacktraces:";
    cpp_dumper.value()([](const std::string& line) { LOG(INFO) << line; });
  }

  if (checkDumpSignal && shouldDump_.load()) {
    // Store debug info to storage if no other thread does it. (By default to
    // local disk)
    std::future<bool> asyncDebugDump = std::async(
        std::launch::async, [this]() { return this->dumpDebuggingInfo(); });

    // wait for the dump until timeout - log data
    waitForFutureOrTimeout(
        asyncDebugDump,
        std::chrono::milliseconds(waitTimeoutDumpInMilSec_),
        "Flight recorder dump in heartbeatMonitor",
        false,
        true);
  }

  if (get_gil_checker() != nullptr) {
    auto fut = launchAsyncGilCheck();
    auto kGilCheckTimeout = std::chrono::milliseconds(300);
    auto futStatus = fut.wait_for(kGilCheckTimeout);
    if (futStatus != std::future_status::ready) {
      TORCH_CHECK(
          futStatus != std::future_status::deferred,
          "Expected the future to have been launched eagerly.");
      LOG(ERROR)
          << "Could not acquire GIL within 300 ms on exit, possible GIL induced hang";
    }
  } else {
    LOG(INFO)
        << "GIL checker was not registered, perhaps this is a no-python build?";
  }

  // There are two possible cases for the watchdog thread exit:
  // Case one: desync report runs quickly, and it follows the step:
  // collective timeout -> desync -> exception handling -> destructors
  // -> set terminateHeartbeatMonitorThread_ -> notify monitorWakeUpCV_.
  // So the code either early returns above or will skip the sleep below.
  // Case two: desync might be slow or get stuck. Or we get stuck in
  // destructors, we will sleep for some time before calling std::abort() to
  // kill the whole process.
  if ((terminateProcessGroup_.load() || collectiveDebugInfoMode_.load() ||
       shouldDump_.load()) &&
      !terminateHeartbeatMonitorThread_.load()) {
    // Leave another two mins for desync report generation or process group
    // destroy.
    std::this_thread::sleep_for(std::chrono::seconds(heartbeatTimeoutInSec_));
    LOG(INFO) << logPrefix() << "slept for " << heartbeatTimeoutInSec_
              << " waiting for desync report or process group destroy.";
  }

  // At this point, we either already sleep for another `heartbeatTimeoutInSec_`
  // or the thread has finished. Because we don't want to block the monitor
  // thread, so We mark the thread detach and the dump of debug info becomes
  // "best effort". If the process exit normally, marking it detach also makes
  // sense because we don't really care about dumping the debug info.

  // We already log completion inside the thread, so it may not be necessary to
  // check the return value here.  We mainly use a future so we can exit early
  // if done.

  if (!terminateHeartbeatMonitorThread_.load()) {
    // Create a error message reported from MonitorThread, so
    // we throw exception and make the whole process to be killed.
    // TODO(fduwjj): After having a hang debug wiki, we need to update the wiki
    // url here.
    if (monitorThreadEnabled_.load()) {
      terminateProcess(getNCCLWatchdogTimeoutExitMsg(exitReason));
    } else {
      // Ideally we want to merge this one with the above one, but we are going
      // to remove the kill switch for monitor thread soon, so we keep this one
      // for now.
      LOG(ERROR)
          << logPrefix()
          << "ProcessGroupNCCL monitor thread is disabled, but would have terminated the process"
          << "after attempting to dump debug info, due to " << exitReason
          << ".";
    }
  }
}

void FlightRecoder::ncclCommWatchdog() {
  c10::setThreadName("pt_nccl_watchdg");

  try {
    VLOG(2) << logPrefix() << "Process group watchdog thread started!";
    ncclHeartbeatMonitorThread_ =
        std::thread(&ProcessGroupNCCL::heartbeatMonitor, this);
    watchdogHandler();
    VLOG(2) << logPrefix()
            << "Process group watchdog thread terminated normally";
  } catch (std::exception& e) {
    if (std::string(e.what()).find("driver shutting down") !=
        std::string::npos) {
      LOG(INFO)
          << logPrefix()
          << "main process destroyed cuda before watchdog loop exited, terminating watchdog."
          << " (Watchdog caught exception: " << e.what();

    } else {
      // Append error message reported from watchdogHandler
      const auto exitMsg = c10::str(
          logPrefix(),
          "Process group watchdog thread terminated with exception: ",
          e.what());
      LOG(ERROR) << exitMsg;
      if (C10_LIKELY(rethrowCUDAErrors_) ||
          !(std::string(e.what()).find("CUDA Error"))) {
        // TODO(whc) clean up the rethrow - why is it stored in a class var and
        // rethrown?
        watchDogException_ =
            std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
        std::rethrow_exception(watchDogException_);
      }
    }
  } catch (...) {
    const auto exitMsg = c10::str(
        logPrefix(),
        "Process group watchdog thread terminated with exception: unknown");
    LOG(ERROR) << exitMsg;
    watchDogException_ =
        std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
    std::rethrow_exception(watchDogException_);
  }
}

void FlightRecoder::logWorkStart(Work& work) {
  if (work.startTraceUpdated_)
    return;

  if (terminateProcessGroup_.load() || storeError_)
    return;

  work.startTraceUpdated_ = true;
  storeError_ = !c10d::traceUpdate(
      store_, traceKeyStart_, work.seq_, opTypeToString(work.opType_));
}

void FlightRecoder::logWorkEnd(Work& work) {
  if (terminateProcessGroup_.load() || storeError_)
    return;

  // In case the start of the work hasn't been logged
  if (!work.startTraceUpdated_) {
    logWorkStart(work);
  }

  storeError_ = !c10d::traceUpdate(
      store_, traceKeyEnd_, work.seq_, opTypeToString(work.opType_));
}

std::string FlightRecoder::getNCCLWatchdogDebugInfo() {
  return retrieveDesyncReport(store_, "NCCL", rank_, size_);
}

bool FlightRecoder::dumpDebuggingInfo() {
  // Serialize all calls to this function to avoid corrupting data, but allow
  // multiple calls in one runtime. User is responsible for preserving the
  // output file from an earlier call before a later call overwrites it.
  static std::mutex writeDebugInfoMutex;
  std::lock_guard<std::mutex> lock(writeDebugInfoMutex);
  LOG(ERROR) << logPrefix() << "ProcessGroupNCCL preparing to dump debug info.";
  if (ncclTraceBufferSize_ > 0) {
    // We dump nccl trace into local disk by default and users can register
    // their customized writer by inheriting `DebugInfoWriter` via
    // `registerDebugInfoWriter`.
    auto ncclTrace = dump_nccl_trace(true, true, false);
    DebugInfoWriter& writer = DebugInfoWriter::getWriter(globalRank());
    LOG(INFO) << logPrefix() << "ProcessGroupNCCL dumping nccl trace to "
              << writer.getWriterTarget();
    writer.write(ncclTrace);
    return true;
  }
  return false;
}