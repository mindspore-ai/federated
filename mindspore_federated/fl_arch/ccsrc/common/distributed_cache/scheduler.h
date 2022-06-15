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
#ifndef MINDSPORE_FL_CACHE_SCHEDULER_H
#define MINDSPORE_FL_CACHE_SCHEDULER_H
#include <string>
#include <map>
#include "distributed_cache/cache_status.h"
#include "distributed_cache/instance_context.h"
namespace mindspore {
namespace fl {
namespace cache {
class Scheduler {
 public:
  static Scheduler &Instance() {
    static Scheduler instance;
    return instance;
  }
  CacheStatus GetInstanceName(const std::string &fl_name, std::string *instance_name);
  CacheStatus GetAllServersRealtime(const std::string &fl_name, std::map<std::string, std::string> *server_map);
  CacheStatus GetAllClusterState(const std::string &fl_name, InstanceState *state);
  CacheStatus SetEnableState(const std::string &fl_name, bool enable);
  CacheStatus StopFLJob(const std::string &fl_name);
  CacheStatus ClearFLJob(const std::string &fl_name);
  CacheStatus OnNewInstance(const std::string &fl_name, const std::string &new_instance_name,
                            const std::string &hyper_params);
  CacheStatus QueryInstance(const std::string &fl_name, std::string *hyper_params);
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_CACHE_SCHEDULER_H
