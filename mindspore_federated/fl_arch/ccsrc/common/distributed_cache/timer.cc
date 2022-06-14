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
#include "distributed_cache/timer.h"
#include <string>
#include <unordered_map>
#include "common/common.h"
#include "distributed_cache/distributed_cache.h"
#include "distributed_cache/redis_keys.h"
#include "server/server.h"
#include "distributed_cache/server.h"
#include "common/fl_context.h"

namespace mindspore {
namespace fl {
namespace cache {
uint64_t Timer::config_expire_time_in_seconds() {
  constexpr uint64_t extra_time_for_config = 7 * 24 * 60 * 60;  // 7 days
  return global_time_window_in_seconds() + extra_time_for_config;
}

uint64_t Timer::iteration_expire_time_in_seconds() {
  constexpr uint64_t extra_time_for_iteration = 30 * 60;  // 30 min
  return global_time_window_in_seconds() + extra_time_for_iteration;
}

uint64_t Timer::release_expire_time_in_seconds() {
  constexpr uint64_t extra_time_for_release = 30;  // 30s
  return extra_time_for_release;
}

uint64_t Timer::global_time_window_in_seconds() {
  constexpr int ms_sec_to_sec = 1000;
  return FLContext::instance()->global_iteration_time_window() / ms_sec_to_sec;
}

// invoked by round handle or other servers notification
CacheStatus Timer::StartTimer(const std::string &name) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = timer_map_.find(name);
  if (it == timer_map_.end()) {
    MS_LOG_WARNING << "Timer " << name << " is not registered";
    return kCacheInnerErr;
  }
  auto &info = it->second;
  if (info.state != kTimerNotStarted) {
    return kCacheSuccess;
  }
  constexpr int sec_to_msec = 1000;
  uint64_t expire_time_in_ms = CURRENT_TIME_MILLI.count() + info.time_in_seconds * sec_to_msec;
  info.state = kTimerStarted;
  info.timeout_stamp = expire_time_in_ms;

  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  auto key = RedisKeys::GetInstance().TimerHash();
  auto ret = client->HSetNx(key, name, std::to_string(info.timeout_stamp));
  if (ret == kCacheExist) {
    ret = client->HGet(key, name, info.timeout_stamp, &info.timeout_stamp);
  }
  client->Expire(key, iteration_expire_time_in_seconds());
  MS_LOG_INFO << "Start timer " << name << ", timer length in seconds: " << info.time_in_seconds
              << ", expire timestamp in milliseconds: " << info.timeout_stamp;
  return ret;
}

CacheStatus Timer::StopTimer(const std::string &name) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = timer_map_.find(name);
  if (it == timer_map_.end()) {
    MS_LOG_WARNING << "Timer " << name << " is not registered";
    return kCacheInnerErr;
  }
  auto &info = it->second;
  info.state = kTimerStopped;
  info.timeout_stamp = 0;
  MS_LOG_INFO << "Stop timer " << name;
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  auto key = RedisKeys::GetInstance().TimerHash();
  return client->HSet(key, name, std::to_string(info.timeout_stamp));
}
//
void Timer::RegisterTimer(const std::string &name, uint64_t time_in_seconds, const Timer::TimerCallback &callback) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = timer_map_.find(name);
  if (it != timer_map_.end()) {
    MS_LOG_WARNING << "Timer " << name << " has already been registered";
    return;
  }
  if (time_in_seconds >= UINT32_MAX) {
    MS_LOG_WARNING << "Duration " << time_in_seconds << " of timer " << name << " cannot >= UINT32_MAX";
    return;
  }
  if (callback == nullptr) {
    MS_LOG_WARNING << "Timeout callback of timer " << name << " cannot be nullptr";
    return;
  }
  auto &item = timer_map_[name];
  item.callback = callback;
  item.time_in_seconds = time_in_seconds;
  MS_LOG_INFO << "Register timer for " << name << ", time length in seconds: " << time_in_seconds;
}

void Timer::ReinitTimer(const std::string &name, uint64_t time_in_seconds) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = timer_map_.find(name);
  if (it == timer_map_.end()) {
    MS_LOG_WARNING << "Timer " << name << " is not registered";
    return;
  }
  if (time_in_seconds >= UINT32_MAX) {
    MS_LOG_WARNING << "Duration " << time_in_seconds << " of timer " << name << " cannot >= UINT32_MAX";
    return;
  }
  auto &info = it->second;
  info.time_in_seconds = time_in_seconds;
  MS_LOG_INFO << "Reinit timer for " << name << ", new time length in seconds: " << time_in_seconds;
}
//
void Timer::ResetOnNewIteration() {
  std::lock_guard<std::mutex> lock(lock_);
  for (auto &item : timer_map_) {
    item.second.state = kTimerNotStarted;
    item.second.timeout_stamp = 0;
  }
  task_que_ = std::queue<TimerCallback>();
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return;
  }
  (void)client->Expire(RedisKeys::GetInstance().TimerHash(), release_expire_time_in_seconds());
}

void Timer::Sync() {
  // if timer is not registered in RedisKeys::GetInstance().TimerHash(), it's stopped or not started
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return;
  }
  std::lock_guard<std::mutex> lock(lock_);
  std::unordered_map<std::string, uint64_t> timer_map;
  auto time_key = RedisKeys::GetInstance().TimerHash();
  auto ret = client->HGetAll(time_key, &timer_map);
  if (!ret.IsSuccess()) {
    MS_LOG_WARNING << "Get timer from cache failed";
    return;
  }
  uint64_t cur_time_in_ms = CURRENT_TIME_MILLI.count();
  auto cur_iteration_num = InstanceContext::Instance().iteration_num();
  std::unordered_map<std::string, uint64_t> timer_reset_map;
  for (auto &item : timer_map_) {
    auto &name = item.first;
    auto &info = item.second;
    if (info.state == kTimerStarted && cur_time_in_ms >= info.timeout_stamp) {
      MS_LOG_INFO << "Timer " << name << " timeout, current time in milliseconds: " << cur_time_in_ms;
      HandleTimeoutInner(&info, cur_iteration_num);  // info.state = kTimerOut
    }
    TimerState cache_state = kTimerNotStarted;
    uint64_t cache_timeout_stamp = 0;
    auto it = timer_map.find(name);
    if (it != timer_map.end()) {
      cache_timeout_stamp = it->second;
      if (cache_timeout_stamp == 0) {
        cache_state = kTimerStopped;
      } else if (cur_time_in_ms >= cache_timeout_stamp) {
        cache_state = kTimerTimeOut;
      } else {
        cache_state = kTimerStarted;
      }
    }
    // The start or stop timer event may be missed, and events need to be synced between cache and local.
    switch (cache_state) {
      case kTimerNotStarted:
        if (info.state != kTimerNotStarted) {          // sync start/stop/timeout event to cache
          timer_reset_map[name] = info.timeout_stamp;  // 0: stopped, other value: started, timeout
        }
        break;
      case kTimerStarted:
        if (info.state == kTimerStopped) {  // sync stop event to cache
          timer_reset_map[name] = info.timeout_stamp;
        } else if (info.state == kTimerNotStarted) {  // sync start event to local
          info.state = cache_state;
          info.timeout_stamp = cache_timeout_stamp;
        }
        break;
      case kTimerStopped:
        if (info.state == kTimerNotStarted || info.state == kTimerStarted) {  // sync stop event to local
          info.state = cache_state;
          info.timeout_stamp = cache_timeout_stamp;
        }
        break;
      case kTimerTimeOut:
        if (info.state == kTimerNotStarted || info.state == kTimerStarted) {
          MS_LOG_INFO << "Timer " << name << " timeout, current time in milliseconds: " << cur_time_in_ms;
          info.timeout_stamp = cache_timeout_stamp;
          HandleTimeoutInner(&info, cur_iteration_num);
        }
        break;
    }
  }
  if (!timer_reset_map.empty()) {
    ret = client->HMSet(time_key, timer_reset_map);
    if (!ret.IsSuccess()) {
      MS_LOG_WARNING << "Failed to sync local event to cache";
      return;
    }
    (void)client->Expire(time_key, iteration_expire_time_in_seconds());
  }
}

void Timer::HandleTimeoutInner(TimerInfo *info, uint64_t event_iteration_num) {
  if (info == nullptr) {
    return;
  }
  if (info->state == kTimerTimeOut || info->state == kTimerStopped) {
    return;
  }
  info->state = kTimerTimeOut;
  SubmitEventHandle(info->callback, event_iteration_num);
}

void Timer::SubmitEventHandle(const TimerCallback &task, uint64_t event_iteration_num) {
  if (task == nullptr) {
    return;
  }
  auto instance_state = InstanceContext::Instance().instance_state();
  if (instance_state != InstanceState::kStateRunning) {
    MS_LOG_INFO << "Instance state is " << GetInstanceStateStr(instance_state) << ", timeout event will not be handled";
    return;
  }
  task_que_.emplace([task, event_iteration_num]() {
    auto cur_iteration_num = InstanceContext::Instance().iteration_num();
    if (event_iteration_num != cur_iteration_num) {
      return;
    }
    task();
  });
  IterationTaskThread::Instance().OnNewTask();
}

bool Timer::HandleEvent() {
  while (true) {
    std::function<void()> task;
    {
      std::lock_guard<std::mutex> lock(lock_);
      if (task_que_.empty()) {
        return true;
      }
      task = task_que_.front();
      task_que_.pop();
    }
    if (task) {
      try {
        task();
      } catch (const std::exception &e) {
        MS_LOG_WARNING << "Catch exception when handle timer callback: " << e.what();
      }
    }
  }
}
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
