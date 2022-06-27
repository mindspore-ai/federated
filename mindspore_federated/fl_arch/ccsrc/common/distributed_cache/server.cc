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
#include "distributed_cache/server.h"
#include "distributed_cache/distributed_cache.h"
#include "distributed_cache/redis_keys.h"
#include "distributed_cache/timer.h"
#include "common/common.h"
#include "common/exit_handler.h"

namespace mindspore {
namespace fl {
namespace cache {
void Server::Init(const std::string &node_id, const std::string &tcp_address) {
  node_id_ = node_id;
  tcp_address_ = tcp_address;
}

void Server::Stop() {
  std::unique_lock<std::mutex> lock(lock_);
  if (!registered_) {
    return;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return;
  }
  // store key
  auto server_key = RedisKeys::GetInstance().ServerHash();
  auto cache_ret = client->HDel(server_key, node_id_);
  if (!cache_ret.IsSuccess()) {
    MS_LOG_WARNING << "Failed to del info of server " << node_id_;
  } else {
    MS_LOG_INFO << "Success to del info of server " << node_id_;
  }
  auto heartbeat_key = RedisKeys::GetInstance().ServerHeartbeatString(node_id_);
  cache_ret = client->Del(heartbeat_key);
  if (!cache_ret.IsSuccess()) {
    MS_LOG_WARNING << "Failed to del heartbeat of server " << node_id_;
  } else {
    MS_LOG_INFO << "Success to del heartbeat of server " << node_id_;
  }
}

std::map<std::string, std::string> Server::GetAllServers() {
  std::unique_lock<std::mutex> lock(lock_);
  if (server_map_.empty()) {
    SyncFromCache2Local();
  }
  return server_map_;
}

CacheStatus Server::GetAllServersRealtime(std::map<std::string, std::string> *server_map) {
  if (server_map == nullptr) {
    return kCacheInnerErr;
  }
  std::unique_lock<std::mutex> lock(lock_);
  auto ret = SyncFromCache2Local();
  if (!ret.IsSuccess()) {
    return ret;
  }
  *server_map = server_map_;
  return kCacheSuccess;
}

CacheStatus Server::SyncFromCache2Local() {
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  std::unordered_map<std::string, std::string> server_registered;
  auto server_key = RedisKeys::GetInstance().ServerHash();
  auto ret = client->HGetAll(server_key, &server_registered);
  if (!ret.IsSuccess()) {
    return ret;
  }
  std::map<std::string, std::string> server_alive;
  for (auto &item : server_registered) {
    const auto &node_id = item.first;
    const auto &node_address = item.second;
    auto heartbeat_key = RedisKeys::GetInstance().ServerHeartbeatString(node_id);
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
  server_map_ = std::move(server_alive);
  return kCacheSuccess;
}

CacheStatus Server::Sync() {
  std::unique_lock<std::mutex> lock(lock_);
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  // store key
  auto server_key = RedisKeys::GetInstance().ServerHash();
  auto cache_ret = client->HSet(server_key, node_id_, tcp_address_);
  if (!cache_ret.IsSuccess()) {
    return cache_ret;
  }
  (void)client->Expire(server_key, Timer::config_expire_time_in_seconds());
  registered_ = true;
  // heartbeat
  auto heartbeat_key = RedisKeys::GetInstance().ServerHeartbeatString(node_id_);
  constexpr uint64_t heartbeat_in_seconds = 10;
  cache_ret = client->SetEx(heartbeat_key, tcp_address_, heartbeat_in_seconds);
  if (!cache_ret.IsSuccess()) {
    return cache_ret;
  }
  // sync other servers
  return SyncFromCache2Local();
}

bool Server::LockCache() {
  MS_LOG_INFO << "Begin to lock server";
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return false;
  }
  auto key = RedisKeys::GetInstance().ServerRegLockString();
  int time_register_timeout = 60 * 15;        // 15 minutes
  constexpr int expire_time_in_seconds = 60;  // lock for 60 seconds
  for (int i = 0; i < time_register_timeout; i++) {
    if (ExitHandler::Instance().HasStopped()) {
      MS_LOG_WARNING << "Lock server failed, the server has receive exit signal "
                     << ExitHandler::Instance().GetSignal();
      return false;
    }
    auto status = client->SetExNx(key, node_id(), expire_time_in_seconds);
    if (status == kCacheExist) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      continue;
    }
    if (!status.IsSuccess()) {
      MS_LOG_WARNING << "Lock server failed, cache server is available or some inner error happened";
      return false;
    }
    own_cache_lock_ = true;
    MS_LOG_INFO << "Lock server successfully";
    return true;
  }
  MS_LOG_WARNING << "Lock server failed, timeout";
  return false;
}

void Server::UnlockCache() {
  if (!own_cache_lock_) {
    return;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return;
  }
  auto key = RedisKeys::GetInstance().ServerRegLockString();
  std::string value;
  auto status = client->Get(key, &value);
  if (status == kCacheNil) {
    MS_LOG_WARNING << "Server lock has expired";
    return;
  }
  if (!status.IsSuccess()) {
    MS_LOG_WARNING << "Failed to unlock server";
    return;
  }
  if (value != node_id()) {
    MS_LOG_WARNING << "Server lock has acquired by other server " << value << ", cur server: " << node_id();
    return;
  }
  status = client->Del(key);
  if (!status.IsSuccess()) {
    MS_LOG_WARNING << "Failed to unlock server";
    return;
  }
  own_cache_lock_ = false;
  MS_LOG_INFO << "Unlock server successfully";
}

CacheStatus Server::Register() { return Sync(); }
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
