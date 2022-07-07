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
#include "distributed_cache/model_info.h"
#include <nlohmann/json.hpp>
#include "common/common.h"
#include "distributed_cache/redis_keys.h"
#include "distributed_cache/timer.h"

namespace mindspore {
namespace fl {
namespace cache {
namespace {
const char *kFeaturesItem = "features";
const char *kFeatureSizeItem = "size";
const char *kFeatureTypeItem = "type";
const char *kFeatureShapeItem = "shape";
const char *kFeatureRequireAggrItem = "require_aggr";
}  // namespace
void ModelInfo::Init(const std::map<std::string, WeightInfo> &weight_infos) { weight_infos_ = weight_infos; }

CacheStatus ModelInfo::SyncPeriod() { return SyncLocal2Cache(weight_infos_); }

CacheStatus ModelInfo::SyncLocal2Cache(const std::map<std::string, WeightInfo> &weight_infos) {
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  nlohmann::json obj;
  auto &weights = obj[kFeaturesItem];
  for (auto &item : weight_infos) {
    auto &weight_info = item.second;
    weights[item.first] = nlohmann::json();
    auto &weight_item = weights[item.first];
    weight_item[kFeatureSizeItem] = weight_info.size;
    weight_item[kFeatureTypeItem] = weight_info.type;
    weight_item[kFeatureRequireAggrItem] = weight_info.require_aggr;
    weight_item[kFeatureShapeItem] = weight_info.shape;
  }
  auto val = obj.dump();
  auto key = RedisKeys::GetInstance().ModelInfoString();
  auto result = client->SetExNx(key, val, Timer::config_expire_time_in_seconds());
  if (result.IsSuccess()) {
    MS_LOG_INFO << "Sync model info to cache success";
  }
  return result;
}

CacheStatus ModelInfo::SyncCache2Local(std::map<std::string, WeightInfo> *weight_infos) {
  if (weight_infos == nullptr) {
    return kCacheInnerErr;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  auto key = RedisKeys::GetInstance().ModelInfoString();
  std::string val;
  auto status = client->Get(key, &val);
  if (!status.IsSuccess()) {
    return status;
  }
  nlohmann::json obj;
  try {
    obj = nlohmann::json::parse(val);
    auto features_it = obj.find(kFeaturesItem);
    if (features_it == obj.end()) {
      MS_LOG_WARNING << "Cannot find " << kFeaturesItem << " in model info json";
      return kCacheTypeErr;
    }
    auto &features_json = features_it.value();
    if (!features_json.is_object()) {
      MS_LOG_WARNING << kFeaturesItem << " in model info json must be json object, but got "
                     << features_json.type_name();
      return kCacheTypeErr;
    }
    std::map<std::string, WeightInfo> infos;
    for (auto &json_item : features_json.get<std::map<std::string, nlohmann::json>>()) {
      WeightInfo info;
      auto &json_val = json_item.second;
      info.name = json_item.first;
      info.shape = json_val[kFeatureShapeItem].get<std::vector<size_t>>();
      info.type = json_val[kFeatureTypeItem];
      info.require_aggr = json_val[kFeatureRequireAggrItem];
      info.size = json_val[kFeatureSizeItem];
      infos[info.name] = info;
    }
    *weight_infos = infos;
    MS_LOG_INFO << "Sync model info from cache success";
  } catch (const std::exception &e) {
    MS_LOG_WARNING << "Failed to parse model info json: " << e.what();
    return kCacheInnerErr;
  }
  return kCacheSuccess;
}
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
