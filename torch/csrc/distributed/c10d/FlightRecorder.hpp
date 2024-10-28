


enum ErrorHandlingMode {
  NoHandling = 0,
  TearDown = 1,
  CleanUpOnly = 2,
  SkipCleanUp = 3
};

#define SHOULD_CLEAN_UP(a) (a != NoHandling && a != SkipCleanUp)

#define SHOULD_TEAR_DOWN(a) (a != NoHandling && a != CleanUpOnly)

class FlightRecoder {

public:
std::string getNCCLWatchdogTimeoutErrorMsg(const std::string& extraMsg);

std::string getNCCLWatchdogTimeoutExitMsg(const std::string& exitReason);

std::unordered_map<std::string, std::unordered_map<std::string, std::string>> ncclDumpMap;


  // Function that runs as part of a separate thread aside from watchdog
  // thread because we need to check the heartbeat from watchdog thread
  // so that when we get stuck in some NCCL/CUDA calls,
  // we can dump the debugging information and abort the process.
  virtual void heartbeatMonitor();

  // Function that directly trigger std::abort so that the whole process
  // gets terminated.
  virtual void terminateProcess(std::string errMsg);

  // When watchdog timeout, this function will be called and return debug info
  // for users. For now we only get information from retrieveDesyncReport.
  // We are working on enabling more useful debug information for watchdog
  // timeout.
  virtual std::string getNCCLWatchdogDebugInfo();

  std::string getNCCLWatchdogTimeoutErrorMsg(const std::string& extraMsg);

  std::string getNCCLWatchdogTimeoutExitMsg(const std::string& exitReason);

  static const int64_t kWatchdogThreadSleepMillis;

  // Watchdog's inside loop.
  // Takes care of cleaning up completed work, and aborting upon failure or
  // timeout.
  void watchdogHandler();


  // Function that runs as part of a separate thread and checks for errors on
  // NCCL communicators. We need a separate thread to check for NCCL errors
  // since we can't rely on the user calling certain methods like wait(),
  // isCompleted() etc. to detect and remediate errors. In addition to this, we
  // need a mechanism to safely abort and remove NCCL communicators from our
  // cache. This can be done cleanly by having a thread for the ProcessGroupNCCL
  // class. Attempting to modify the communicator cache from the WorkNCCL class
  // might run into issues with object lifetime since the ProcessGroupNCCL
  // object might get destroyed before the WorkNCCL object.
  void ncclCommWatchdog();

  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminateProcessGroup_;


  // Heartbeat of watchdog thread.
  std::atomic_uint64_t heartbeat_;

  // The time interval used for deciding whether there is no watchdog heartbeat.
  int heartbeatTimeoutInSec_;

  // timeout for the dump to finish.
  int waitTimeoutDumpInMilSec_;

  // Interval of check coordinated signals in ProcessGroupNCCL from other ranks
  // e.g., trigger the dump of the debugging info for timeout when notified.
  int coordCheckIntervalMilSec_;

  // Size of ring buffer where we store NCCL Traces for debugging.
  int ncclTraceBufferSize_;

  // We gate the heartbeat monitor thread so that we can roll it out gradually.
  std::atomic<bool> monitorThreadEnabled_;

  // We gate the cudaEventCache so that we can roll it out gradually.
  std::atomic<bool> cudaEventCacheEnabled_;

  // Monitor thread which checks the heartbeat of Watchdog thread.
  // If the monitor thread finds there is no heartbeat, it will dump debug info
  // and then kill the watchdog thread to avoid hang.
  std::thread ncclHeartbeatMonitorThread_;

  // Watchdog thread which looks for errors on the cached NCCL communicators.
  std::thread ncclCommWatchdogThread_;

  std::thread onCompletionHookThread_;

  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminateProcessGroup_;

  // Whether or not we should terminate the heartbeat monitoring threads.
  std::atomic<bool> terminateHeartbeatMonitorThread_;

  // Whether we are in the shutdown mode when we are trying to get debug info,
  // such as desync report.
  std::atomic<bool> collectiveDebugInfoMode_;

  // Whether there are hooks pending to be fired
  std::atomic<bool> hasPendingHooks_;

  // This is the signal from watchdog threads to indicate whether the monitor
  // thread should dump. Making it static so that it is accessiable from all the
  // PGs. With this flag, monitor thread would dump debug info under any one of
  // the three conditions:
  //
  // 1: watchdog thread of any PG detects a collective timeout.
  // 2: timeout signal is received from other ranks through tcpstore.
  // 3: current PG's watchdog heartbeat timeout occurs.
  //
  // Note that only the monitor thread from PG0 will dump the debug info for
  // case one and two so that the debug info is only dumped once.
  static std::atomic<bool> shouldDump_;

  // Mutex to Guard workMetaList_
  std::mutex workMetaListMutex_;

  // Mutex to Guard monitorWakeUpCV_
  std::mutex monitorMutex_;

  bool writeDebugInfo_ = false;

  // Condition Variable for watchdog thread sleep
  std::condition_variable workMetaListCV_;

  // Condition Variable for monitor thread to wake up early
  std::condition_variable monitorWakeUpCV_;

  // Vector to Store WorkNCCL pointers
  std::list<ProcessGroupNCCL::WorkNCCL> workMetaList_;

  std::chrono::time_point<std::chrono::steady_clock> lastWorkListUpdateTime_;

  // Mutex to Guard workMetaList_
  std::mutex completedWorkListMutex_;

  // Condition Variable for watchdog thread sleep
  std::condition_variable completedWorkListCV_;

  std::list<ProcessGroupNCCL::WorkNCCL> completedWorkList_;

  std::shared_ptr<ProcessGroupStatus> pgStatus_;

    // Whether or not to sleep after an exception is thrown in the watchdog.
  bool sleepAfterException_;

  // Whether or not to enable timeout root cause analysis.
  bool desyncDebug_;

  // Whether or not the workCleanupThread is used to perform async error
  // handling.
  ErrorHandlingMode asyncErrorHandling_ = NoHandling;

  // Desync debug helper
  void logWorkStart(Work& work);

  // Desync debug helper
  void logWorkEnd(Work& work);

}