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
#ifndef MINDSPORE_FL_CACHE_MODEL_H
#define MINDSPORE_FL_CACHE_MODEL_H
#include <string>
#include <vector>
#include <map>
#include "distributed_cache/distributed_cache.h"

namespace mindspore {
namespace fl {
namespace cache {
struct WeightInfo {
  std::string name;
  size_t size = 0;
  std::vector<size_t> shape;  // no check
  std::string type;
  bool require_aggr = true;
};

class ModelInfo {
 public:
  static ModelInfo &Instance() {
    static ModelInfo instance;
    return instance;
  }
  void Init(const std::map<std::string, WeightInfo> &weight_infos);

  CacheStatus SyncPeriod();

  static CacheStatus SyncLocal2Cache(const std::map<std::string, WeightInfo> &weight_infos);
  static CacheStatus SyncCache2Local(std::map<std::string, WeightInfo> *weight_infos);

 private:
  std::map<std::string, WeightInfo> weight_infos_;
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_CACHE_MODEL_H
