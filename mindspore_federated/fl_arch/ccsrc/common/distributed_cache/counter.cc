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
#include "distributed_cache/counter.h"
#include <memory>
#include "common/common.h"
#include "distributed_cache/distributed_cache.h"
#include "distributed_cache/redis_keys.h"
#include "server/server.h"
#include "distributed_cache/server.h"
#include "distributed_cache/timer.h"
#include "distributed_cache/iteration_task_thread.h"
namespace mindspore {
namespace fl {
namespace cache {
void Counter::RegisterCounter(const std::string &name, uint64_t threshold,
                              const Counter::CounterCallback &first_callback,
                              const Counter::CounterCallback &last_callback) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = counter_map_.find(name);
  if (it != counter_map_.end()) {
    MS_LOG_WARNING << "Count " << name << " has already been registered";
    return;
  }
  if (threshold > UINT32_MAX) {
    MS_LOG_WARNING << "Threshold " << threshold << " of count " << name << " cannot >= UINT32_MAX";
    return;
  }
  auto &item = counter_map_[name];
  item.first_callback = first_callback;
  item.last_callback = last_callback;
  item.server_hash_ = false;
  item.threshold = threshold;
  MS_LOG_INFO << "Register counter for " << name << ", threshold: " << threshold;
}

void Counter::RegisterPerServerCounter(const std::string &name, uint64_t threshold,
                                       const CounterCallback &first_callback, const CounterCallback &last_callback) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = counter_map_.find(name);
  if (it != counter_map_.end()) {
    MS_LOG_WARNING << "Count " << name << " has already been registered";
    return;
  }
  if (threshold > UINT32_MAX) {
    MS_LOG_WARNING << "Threshold " << threshold << " of count " << name << " cannot >= UINT32_MAX";
    return;
  }
  auto &item = counter_map_[name];
  item.first_callback = first_callback;
  item.last_callback = last_callback;
  item.server_hash_ = true;
  item.threshold = threshold;
  MS_LOG_INFO << "Register counter(total count of all valid servers) for " << name << ", threshold: " << threshold;
}

void Counter::ReinitCounter(const std::string &name, uint64_t threshold) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = counter_map_.find(name);
  if (it == counter_map_.end()) {
    MS_LOG_WARNING << "Cannot find count " << name << " registered";
    return;
  }
  if (threshold <= 0) {
    MS_LOG_WARNING << "Threshold " << threshold << " of count " << name << " cannot <= 0";
    return;
  }
  auto &info = it->second;
  info.threshold = threshold;
  MS_LOG_INFO << "Reinit counter for " << name << ", new threshold: " << threshold;
}

void Counter::ResetOnNewIteration() {
  std::lock_guard<std::mutex> lock(lock_);
  for (auto &item : counter_map_) {
    item.second.first_triggered = false;
    item.second.last_triggered = false;
    item.second.has_server_exit = false;
  }
  task_que_ = std::queue<CounterCallback>();
  // for expire and release
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return;
  }
  auto rel_time_in_seconds = Timer::release_expire_time_in_seconds();
  (void)client->Expire(RedisKeys::GetInstance().CountHash(), rel_time_in_seconds);
  for (auto &item : counter_map_) {
    if (item.second.server_hash_) {
      auto server_hash_key = RedisKeys::GetInstance().CountPerServerHash(item.first);
      (void)client->Expire(server_hash_key, rel_time_in_seconds);
    }
  }
}

bool Counter::ReachThreshold(const std::string &name) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = counter_map_.find(name);
  if (it == counter_map_.end()) {
    MS_LOG_WARNING << "Cannot find count " << name << " registered";
    return true;
  }
  auto &info = it->second;
  if (info.last_triggered) {
    return true;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return true;
  }
  uint64_t count = 0;
  if (!GetCountInner(client, name, &count)) {
    return true;
  }
  return count >= info.threshold;
}

bool Counter::Count(const std::string &name, bool *trigger_first, bool *trigger_last) {
  if (trigger_first == nullptr || trigger_last == nullptr) {
    return false;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return false;
  }
  std::lock_guard<std::mutex> lock(lock_);
  auto it = counter_map_.find(name);
  if (it == counter_map_.end()) {
    MS_LOG_WARNING << "Cannot find count " << name << " registered";
    return false;
  }
  auto cur_iteration_num = InstanceContext::Instance().iteration_num();
  auto &info = it->second;
  uint64_t new_count = 0;
  if (info.server_hash_) {
    auto key = RedisKeys::GetInstance().CountPerServerHash(name);
    uint64_t temp_value = 0;
    auto ret = client->HIncr(key, Server::Instance().node_id(), &temp_value);
    if (!ret.IsSuccess()) {
      MS_LOG_WARNING << "Get hash count " << name << " failed";
      return false;
    }
    if (temp_value == 1) {
      (void)client->Expire(key, Timer::iteration_expire_time_in_seconds());
    }
    if (!GetCountInner(client, name, &new_count)) {
      MS_LOG_WARNING << "Get hash count " << name << " failed";
      return false;
    }
  } else {
    auto key = RedisKeys::GetInstance().CountHash();
    auto ret = client->HIncr(key, name, &new_count);
    if (!ret.IsSuccess()) {
      MS_LOG_WARNING << "Incr string count " << name << " failed";
      return false;
    }
    if (new_count == 1) {
      (void)client->Expire(key, Timer::iteration_expire_time_in_seconds());
    }
  }
  *trigger_first = (new_count == 1);
  *trigger_last = (new_count == info.threshold);
  if (new_count >= 1 && !info.first_triggered) {
    HandleFirstCountEvent(&info, cur_iteration_num);
  }
  if (new_count >= info.threshold && !info.last_triggered) {
    HandleLastCountEvent(&info, cur_iteration_num);
  }
  return true;
}

bool Counter::HasServerExit(const std::string &name) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = counter_map_.find(name);
  if (it == counter_map_.end()) {
    MS_LOG_WARNING << "Cannot find count " << name << " registered";
    return false;
  }
  auto &item = it->second;
  return item.server_hash_ && item.has_server_exit;
}

void Counter::OnNotifyCountEvent(const ServerBroadcastMessage &msg) {
  std::lock_guard<std::mutex> lock(lock_);
  const auto &name = msg.count_name();
  auto it = counter_map_.find(name);
  if (it == counter_map_.end()) {
    MS_LOG_WARNING << "Cannot find count " << name << " registered";
    return;
  }
  auto &item = it->second;
  if (msg.trigger_first()) {
    HandleFirstCountEvent(&item, msg.cur_iteration_num());
  }
  if (msg.trigger_last()) {
    HandleLastCountEvent(&item, msg.cur_iteration_num());
  }
}

void Counter::HandleFirstCountEvent(CounterInfo *info, uint64_t event_iteration_num) {
  if (info == nullptr) {
    return;
  }
  if (info->first_triggered) {
    return;
  }
  info->first_triggered = true;
  SubmitEventHandle(info->first_callback, event_iteration_num);
}

void Counter::HandleLastCountEvent(CounterInfo *info, uint64_t event_iteration_num) {
  if (info == nullptr) {
    return;
  }
  if (info->last_triggered) {
    return;
  }
  info->last_triggered = true;
  SubmitEventHandle(info->last_callback, event_iteration_num);
}

void Counter::SubmitEventHandle(const CounterCallback &task, uint64_t event_iteration_num) {
  if (task == nullptr) {
    return;
  }
  auto instance_state = InstanceContext::Instance().instance_state();
  if (instance_state != InstanceState::kStateRunning) {
    MS_LOG_INFO << "Instance state is " << GetInstanceStateStr(instance_state) << ", count event will not be handled";
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

bool Counter::HandleEvent() {
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
        MS_LOG_WARNING << "Catch exception when handle counter callback: " << e.what();
      }
    }
  }
}

bool Counter::GetCountInner(const std::shared_ptr<RedisClientBase> &client, const std::string &name, uint64_t *count) {
  if (client == nullptr || count == nullptr) {
    return false;
  }
  auto it = counter_map_.find(name);
  if (it == counter_map_.end()) {
    return false;
  }
  auto &info = it->second;
  uint64_t cur_count = 0;
  if (info.server_hash_) {
    auto server_map = cache::Server::Instance().GetAllServers();
    if (server_map.empty()) {
      MS_LOG_WARNING << "Get servers from cache failed";
      return false;
    }
    auto key = RedisKeys::GetInstance().CountPerServerHash(name);
    std::unordered_map<std::string, uint64_t> count_map;
    auto ret = client->HGetAll(key, &count_map);
    if (!ret.IsSuccess()) {
      MS_LOG_WARNING << "Get hash count " << name << " failed";
      return false;
    }
    bool has_server_exit = false;
    for (auto &count_item : count_map) {
      if (!server_map.count(count_item.first)) {
        has_server_exit = true;
      }
      cur_count += count_item.second;
    }
    info.has_server_exit = has_server_exit;
  } else {
    auto key = RedisKeys::GetInstance().CountHash();
    auto ret = client->HGet(key, name, 0, &cur_count);
    if (!ret.IsSuccess()) {
      MS_LOG_WARNING << "Get string count " << name << " failed";
      return false;
    }
  }
  *count = cur_count;
  return true;
}

CacheStatus Counter::GetPerServerCountMap(const std::string &name,
                                          std::unordered_map<std::string, uint64_t> *count_map) {
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  auto key = RedisKeys::GetInstance().CountPerServerHash(name);
  auto ret = client->HGetAll(key, count_map);
  if (!ret.IsSuccess()) {
    MS_LOG_WARNING << "Get hash count " << name << " failed";
    return ret;
  }
  return kCacheSuccess;
}

void Counter::Sync() {
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return;
  }
  std::lock_guard<std::mutex> lock(lock_);
  std::unordered_map<std::string, uint64_t> count_map;
  auto count_key = RedisKeys::GetInstance().CountHash();
  auto ret = client->HGetAll(count_key, &count_map);
  if (!ret.IsSuccess()) {
    MS_LOG_WARNING << "Get count failed, skip sync";
    return;
  }
  auto cur_iteration_num = InstanceContext::Instance().iteration_num();
  for (auto &item : counter_map_) {
    const auto &name = item.first;
    auto &info = item.second;
    uint64_t cur_count = 0;
    if (info.server_hash_) {
      if (!GetCountInner(client, name, &cur_count)) {
        MS_LOG_WARNING << "Get count of " << name << ", skip sync";
        continue;
      }
    } else {
      cur_count = count_map[name];
    }
    if (cur_count >= 1 && !info.first_triggered) {
      HandleFirstCountEvent(&info, cur_iteration_num);
    }
    if (cur_count >= info.threshold && !info.last_triggered) {
      HandleLastCountEvent(&info, cur_iteration_num);
    }
  }
  auto expire_time_in_seconds = Timer::iteration_expire_time_in_seconds();
  (void)client->Expire(RedisKeys::GetInstance().CountHash(), expire_time_in_seconds);
  for (auto &item : counter_map_) {
    if (item.second.server_hash_) {
      auto server_hash_key = RedisKeys::GetInstance().CountPerServerHash(item.first);
      (void)client->Expire(server_hash_key, expire_time_in_seconds);
    }
  }
}
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
