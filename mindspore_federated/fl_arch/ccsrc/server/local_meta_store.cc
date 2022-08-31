/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "server/local_meta_store.h"

namespace mindspore {
namespace fl {
namespace server {
void LocalMetaStore::remove_value(const std::string &name) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (key_to_meta_.count(name) != 0) {
    (void)key_to_meta_.erase(key_to_meta_.find(name));
  }
}

bool LocalMetaStore::has_value(const std::string &name) {
  std::unique_lock<std::mutex> lock(mtx_);
  return key_to_meta_.count(name) != 0;
}

void LocalMetaStore::set_curr_iter_num(size_t num) {
  std::unique_lock<std::mutex> lock(mtx_);
  curr_iter_num_ = num;
}

const size_t LocalMetaStore::curr_iter_num() {
  std::unique_lock<std::mutex> lock(mtx_);
  return curr_iter_num_;
}

void LocalMetaStore::set_curr_instance_state(cache::InstanceState instance_state) { instance_state_ = instance_state; }

const cache::InstanceState LocalMetaStore::curr_instance_state() { return instance_state_; }

const void LocalMetaStore::put_aggregation_feature_map(ModelItemPtr modelItemPtr) {
  aggregation_feature_map_ = modelItemPtr;
}

ModelItemPtr &LocalMetaStore::aggregation_feature_map() { return aggregation_feature_map_; }

bool LocalMetaStore::verifyAggregationFeatureMap(const ModelItemPtr &modelItemPtr) {
  // feature map size in Hybrid training is not equal with upload model size
  if (modelItemPtr->weight_items.size() > aggregation_feature_map_->weight_items.size()) {
    return false;
  }

  for (const auto &weight : modelItemPtr->weight_items) {
    std::string weight_name = weight.first;
    size_t weight_size = weight.second.size;

    if (aggregation_feature_map_->weight_items.count(weight_name) == 0) {
      return false;
    }
    if (weight_size != aggregation_feature_map_->weight_items[weight_name].size) {
      return false;
    }
  }
  float *data_arr = reinterpret_cast<float *>(modelItemPtr->weight_data.data());
  std::vector<float> weight_data(data_arr, data_arr + modelItemPtr->weight_data.size() / sizeof(float));

  for (const auto &data : weight_data) {
    if (std::isnan(data) || std::isinf(data)) {
      MS_LOG(WARNING) << "The aggregation weight is nan or inf.";
      return false;
    }
  }
  return true;
}

bool LocalMetaStore::verifyAggregationFeatureMap(const std::map<std::string, Address> &model) {
  ModelItemPtr modelItemPtr = std::make_shared<ModelItem>();
  for (const auto &item : model) {
    WeightItem weight;
    std::string weight_full_name = item.first;
    size_t weight_size = item.second.size;

    weight.name = weight_full_name;
    weight.size = weight_size;

    uint8_t *upload_weight_data_arr = reinterpret_cast<uint8_t *>(const_cast<void *>(item.second.addr));
    std::vector<uint8_t> upload_weight_data(upload_weight_data_arr,
                                            upload_weight_data_arr + weight_size / sizeof(uint8_t));
    modelItemPtr->weight_items[weight_full_name] = weight;
    modelItemPtr->weight_data.insert(modelItemPtr->weight_data.end(), upload_weight_data.begin(),
                                     upload_weight_data.end());
  }
  return verifyAggregationFeatureMap(modelItemPtr);
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
