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
#include "distributed_cache/worker.h"
#include "distributed_cache/distributed_cache.h"
#include "distributed_cache/redis_keys.h"
#include "distributed_cache/timer.h"
#include "distributed_cache/scheduler.h"
#include "common/common.h"
namespace mindspore {
namespace fl {
namespace cache {
void Worker::Init(const std::string &node_id, const std::string &fl_name) {
  fl_id_ = node_id;
  fl_name_ = fl_name;
}

void Worker::Stop() {
  std::unique_lock<std::mutex> lock(lock_);
  if (!registered_) {
    return;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return;
  }
  std::string instance_name;
  (void)Scheduler::Instance().GetInstanceName(fl_name_, &instance_name);
  if (instance_name.empty()) {
    return;
  }
  // store key
  auto worker_key = RedisKeys::GetInstance().WorkerHash(fl_name_, instance_name);
  auto cache_ret = client->HDel(worker_key, fl_id_);
  if (!cache_ret.IsSuccess()) {
    MS_LOG_WARNING << "Failed to del info of worker " << fl_id_;
  } else {
    MS_LOG_INFO << "Success to del info of worker " << fl_id_;
  }
  auto heartbeat_key = RedisKeys::GetInstance().WorkerHeartbeatString(fl_name_, instance_name, fl_id_);
  cache_ret = client->Del(heartbeat_key);
  if (!cache_ret.IsSuccess()) {
    MS_LOG_WARNING << "Failed to del heartbeat of worker " << fl_id_;
  } else {
    MS_LOG_INFO << "Success to del heartbeat of worker " << fl_id_;
  }
}

CacheStatus Worker::SyncFromCache2Local() {
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  std::string instance_name;
  auto ret = Scheduler::Instance().GetInstanceName(fl_name_, &instance_name);
  if (instance_name.empty()) {
    return ret;
  }
  std::unordered_map<std::string, std::string> worker_registered;
  auto worker_key = RedisKeys::GetInstance().WorkerHash(fl_name_, instance_name);
  ret = client->HGetAll(worker_key, &worker_registered);
  if (!ret.IsSuccess()) {
    return ret;
  }
  for (auto &item : worker_registered) {
    const auto &node_id = item.first;
    auto heartbeat_key = RedisKeys::GetInstance().WorkerHeartbeatString(fl_name_, instance_name, node_id);
    std::string temp_val;
    ret = client->Get(heartbeat_key, &temp_val);
    if (ret == kCacheNil) {
      MS_LOG_WARNING << "Worker " << node_id << " heartbeat timeout";
      (void)client->HDel(worker_key, node_id);
      continue;
    }
    if (!ret.IsSuccess()) {
      return ret;
    }
  }
  return kCacheSuccess;
}

CacheStatus Worker::Sync() {
  std::unique_lock<std::mutex> lock(lock_);
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  std::string instance_name;
  auto ret = Scheduler::Instance().GetInstanceName(fl_name_, &instance_name);
  if (instance_name.empty()) {
    return ret;
  }
  // store key
  auto worker_key = RedisKeys::GetInstance().WorkerHash(fl_name_, instance_name);
  auto cache_ret = client->HSet(worker_key, fl_id_, fl_id_);
  if (!cache_ret.IsSuccess()) {
    return cache_ret;
  }
  (void)client->Expire(worker_key, Timer::config_expire_time_in_seconds());
  registered_ = true;
  // heartbeat
  auto heartbeat_key = RedisKeys::GetInstance().WorkerHeartbeatString(fl_name_, instance_name, fl_id_);
  constexpr uint64_t heartbeat_in_seconds = 10;
  cache_ret = client->SetEx(heartbeat_key, fl_id_, heartbeat_in_seconds);
  if (!cache_ret.IsSuccess()) {
    return cache_ret;
  }
  return SyncFromCache2Local();
}

CacheStatus Worker::Register() { return Sync(); }
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
