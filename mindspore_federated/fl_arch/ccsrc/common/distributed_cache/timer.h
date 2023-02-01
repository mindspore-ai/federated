/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_CCSRC_FL_CACHE_TIMER_H
#define MINDSPORE_CCSRC_FL_CACHE_TIMER_H
#include <string>
#include <unordered_map>
#include <functional>
#include <queue>
#include "common/protos/comm.pb.h"
#include "distributed_cache/distributed_cache.h"

namespace mindspore {
namespace fl {
namespace cache {
class Timer {
 public:
  static Timer &Instance() {
    static Timer instance;
    return instance;
  }
  void SyncWithCache();
  void Sync();
  void CacheStateHandler(std::unordered_map<std::string, uint64_t> *timer_reset_map,
                         std::unordered_map<std::string, uint64_t> *timer_map);
  // invoked by round handle or other servers notification
  CacheStatus StartTimerWithCache(const std::string &name);
  CacheStatus StopTimerWithCache(const std::string &name);

  CacheStatus StartTimer(const std::string &name);
  CacheStatus StopTimer(const std::string &name);
  //
  using TimerCallback = std::function<void()>;
  void RegisterTimer(const std::string &name, uint64_t time_in_seconds, const TimerCallback &callback);
  void ReinitTimer(const std::string &name, uint64_t time_in_seconds);
  //
  void ResetOnNewIteration();

  static uint64_t config_expire_time_in_seconds();
  static uint64_t iteration_expire_time_in_seconds();
  static uint64_t release_expire_time_in_seconds();
  static uint64_t global_time_window_in_seconds();
  static uint64_t unsupervised_data_expire_time_in_seconds();
  bool HandleEvent();

 private:
  // for local state
  enum TimerState {
    kTimerNotStarted = 0,
    kTimerStarted = 1,
    kTimerStopped = 2,
    kTimerTimeOut = 3,
  };
  // for cache state
  // TimerNotStarted: do not have field
  // TimerStarted and TimerTimeOut: has timeout_stamp value != 0
  // TimerStopped: has timeout_stamp == 0
  struct TimerInfo {
    uint64_t time_in_seconds = 0;
    TimerCallback callback = nullptr;
    TimerState state = kTimerNotStarted;
    uint64_t timeout_stamp = 0;  // timeout stamp
  };
  std::unordered_map<std::string, TimerInfo> timer_map_;
  std::mutex lock_;
  std::queue<TimerCallback> task_que_;
  void HandleTimeoutInner(TimerInfo *info, uint64_t event_iteration_num);

  void SubmitEventHandle(const TimerCallback &task, uint64_t event_iteration_num);
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_CACHE_TIMER_H
