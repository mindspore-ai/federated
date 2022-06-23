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
#ifndef MINDSPORE_FL_DISTRIBUTED_CACHE_WORKER_H
#define MINDSPORE_FL_DISTRIBUTED_CACHE_WORKER_H
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
class Worker {
 public:
  static Worker &Instance() {
    static Worker instance;
    return instance;
  }
  CacheStatus Sync();

  void Init(const std::string &fl_id, const std::string &fl_name);
  void Stop();

  std::string fl_id() const { return fl_id_; }
  CacheStatus Register();

 private:
  std::string fl_id_;
  std::string fl_name_;
  std::mutex lock_;
  bool registered_ = false;

  CacheStatus SyncFromCache2Local();
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_DISTRIBUTED_CACHE_WORKER_H
