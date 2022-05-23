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

#ifndef MINDSPORE_CCSRC_FL_LOAD_CONFIG_H
#define MINDSPORE_CCSRC_FL_LOAD_CONFIG_H

#include <string>
#include "distributed_cache/distributed_cache.h"

namespace mindspore {
namespace fl {
namespace cache {
class HyperParams {
 public:
  static HyperParams &Instance() {
    static HyperParams instance;
    return instance;
  }
  CacheStatus InitAndSync();

  // on init and new instance
  CacheStatus SyncOnNewInstance();
  CacheStatus SyncPeriod();

  static bool MergeHyperJsonConfig(const std::string &fl_name, const std::string &hyper_params, std::string *error_msg,
                                   std::string *output_hyper_params);

 private:
  CacheStatus SyncLocal2Cache(const std::shared_ptr<RedisClientBase> &client);
  CacheStatus SyncCache2Local(const std::shared_ptr<RedisClientBase> &client);
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_LOAD_CONFIG_H
