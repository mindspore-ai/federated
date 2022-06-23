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
#include "distributed_cache/scheduler.h"
#include "distributed_cache/distributed_cache.h"
#include "common/common.h"
#include "distributed_cache/common.h"
#include "distributed_cache/timer.h"
#include "distributed_cache/redis_keys.h"

namespace mindspore {
namespace fl {
namespace cache {
namespace {
const char *kFiledRunningState = "runningState";
}  // namespace
CacheStatus Scheduler::GetInstanceName(const std::string &fl_name, std::string *instance_name) {
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  auto key = RedisKeys::GetInstance().InstanceNameString(fl_name);
  return client->Get(key, instance_name);
}

CacheStatus Scheduler::GetAllServersRealtime(const std::string &fl_name,
                                             std::map<std::string, std::string> *server_map) {
  std::string instance_name;
  auto ret = GetInstanceName(fl_name, &instance_name);
  if (instance_name.empty()) {
    return ret;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  std::unordered_map<std::string, std::string> server_registered;
  auto server_key = RedisKeys::GetInstance().ServerHash(fl_name, instance_name);
  ret = client->HGetAll(server_key, &server_registered);
  if (!ret.IsSuccess()) {
    return ret;
  }
  std::map<std::string, std::string> server_alive;
  for (auto &item : server_registered) {
    const auto &node_id = item.first;
    const auto &node_address = item.second;
    auto heartbeat_key = RedisKeys::GetInstance().ServerHeartbeatString(fl_name, instance_name, node_id);
    std::string temp_val;
    ret = client->Get(heartbeat_key, &temp_val);
    if (ret == kCacheNil) {
      MS_LOG_WARNING << "Server " << node_id << " heartbeat timeout";
      (void)client->HDel(server_key, node_id);
      continue;
    }
    if (!ret.IsSuccess()) {
      return ret;
    }
    server_alive[node_id] = node_address;
  }
  *server_map = std::move(server_alive);
  return kCacheSuccess;
}

CacheStatus Scheduler::GetAllWorkersRealtime(const std::string &fl_name,
                                             std::map<std::string, std::string> *worker_map) {
  std::string instance_name;
  auto ret = GetInstanceName(fl_name, &instance_name);
  if (instance_name.empty()) {
    return ret;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  std::unordered_map<std::string, std::string> worker_registered;
  auto worker_key = RedisKeys::GetInstance().WorkerHash(fl_name, instance_name);
  ret = client->HGetAll(worker_key, &worker_registered);
  if (!ret.IsSuccess()) {
    return ret;
  }
  std::map<std::string, std::string> worker_alive;
  for (auto &item : worker_registered) {
    const auto &node_id = item.first;
    const auto &node_address = item.second;
    auto heartbeat_key = RedisKeys::GetInstance().WorkerHeartbeatString(fl_name, instance_name, node_id);
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
    worker_alive[node_id] = node_address;
  }
  *worker_map = std::move(worker_alive);
  return kCacheSuccess;
}

CacheStatus Scheduler::GetAllClusterState(const std::string &fl_name, InstanceState *state) {
  std::string instance_name;
  auto ret = GetInstanceName(fl_name, &instance_name);
  if (instance_name.empty()) {
    return ret;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  auto key = RedisKeys::GetInstance().InstanceStatusHash(fl_name, instance_name);

  auto instance_state_cache = static_cast<uint64_t>(InstanceState::kStateRunning);
  ret = client->HGet(key, kFiledRunningState, instance_state_cache, &instance_state_cache);
  if (!ret.IsSuccess()) {
    MS_LOG_WARNING << "Get iteration state from distributed buffer failed";
    return ret;
  }
  *state = static_cast<InstanceState>(instance_state_cache);
  return kCacheSuccess;
}

CacheStatus Scheduler::SetEnableState(const std::string &fl_name, bool enable) {
  std::string instance_name;
  auto ret = GetInstanceName(fl_name, &instance_name);
  if (instance_name.empty()) {
    return ret;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  auto key = RedisKeys::GetInstance().InstanceStatusHash(fl_name, instance_name);
  InstanceState state_new;
  if (enable) {
    state_new = kStateRunning;
  } else {
    state_new = kStateDisable;
  }
  return client->HSet(key, kFiledRunningState, std::to_string(static_cast<int>(state_new)));
}

CacheStatus Scheduler::StopFLJob(const std::string &fl_name) {
  std::string instance_name;
  auto ret = GetInstanceName(fl_name, &instance_name);
  if (instance_name.empty()) {
    return ret;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  auto key = RedisKeys::GetInstance().InstanceStatusHash(fl_name, instance_name);
  InstanceState state_new = kStateStop;
  return client->HSet(key, kFiledRunningState, std::to_string(static_cast<int>(state_new)));
}

CacheStatus Scheduler::ClearFLJob(const std::string &fl_name) {
  std::string instance_name;
  auto ret = GetInstanceName(fl_name, &instance_name);
  if (instance_name.empty()) {
    return ret;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  std::vector<std::string> del_keys = {
    RedisKeys::GetInstance().InstanceNameString(fl_name),
    RedisKeys::GetInstance().InstanceStatusHash(fl_name, instance_name),
    RedisKeys::GetInstance().HyperParamsString(fl_name, instance_name),
    RedisKeys::GetInstance().ServerHash(fl_name, instance_name),
  };
  return client->Del(del_keys);
}

CacheStatus Scheduler::OnNewInstance(const std::string &fl_name, const std::string &new_instance_name,
                                     const std::string &hyper_params) {
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  {  // set new instance config
    auto key = RedisKeys::GetInstance().HyperParamsString(fl_name, new_instance_name);
    auto result = client->SetEx(key, hyper_params, Timer::config_expire_time_in_seconds());
    if (!result.IsSuccess()) {
      return result;
    }
  }
  {  // current instance set new instance
    auto key = RedisKeys::GetInstance().InstanceNameString(fl_name);
    auto status = client->SetEx(key, new_instance_name, Timer::config_expire_time_in_seconds());
    return status;  // failed, exist or success
  }
}

CacheStatus Scheduler::QueryInstance(const std::string &fl_name, std::string *hyper_params) {
  if (hyper_params == nullptr) {
    return kCacheInnerErr;
  }
  std::string instance_name;
  auto ret = GetInstanceName(fl_name, &instance_name);
  if (instance_name.empty()) {
    return ret;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  auto key = RedisKeys::GetInstance().HyperParamsString(fl_name, instance_name);
  auto result = client->Get(key, hyper_params);
  return result;
}
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
