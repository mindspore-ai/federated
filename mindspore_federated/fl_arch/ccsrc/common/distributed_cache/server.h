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
#ifndef MINDSPORE_FL_DISTRIBUTED_CACHE_SERVER_H
#define MINDSPORE_FL_DISTRIBUTED_CACHE_SERVER_H
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <functional>
#include <mutex>
#include "distributed_cache/iteration_task_thread.h"
#include "distributed_cache/cache_status.h"
namespace mindspore {
namespace fl {
namespace cache {
class Server {
 public:
  static Server &Instance() {
    static Server instance;
    return instance;
  }
  CacheStatus Sync();
  std::map<std::string, std::string> GetAllServers();
  CacheStatus GetAllServersRealtime(std::map<std::string, std::string> *server_map);

  void Init(const std::string &node_id, const std::string &tcp_address);
  void Stop();

  std::string node_id() const { return node_id_; }
  std::string tcp_address() const { return tcp_address_; }

  bool LockCache();
  void UnlockCache();
  CacheStatus Register();

 private:
  std::string node_id_;
  std::string tcp_address_;
  std::map<std::string, std::string> server_map_;
  std::mutex lock_;
  bool own_cache_lock_ = false;
  bool registered_ = false;

  CacheStatus SyncFromCache2Local();
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_DISTRIBUTED_CACHE_SERVER_H
