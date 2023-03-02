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

#include "server/model_store.h"
#include <map>
#include <string>
#include <memory>
#include <utility>
#include "common/utils/python_adapter.h"
#include "distributed_cache/instance_context.h"

namespace mindspore {
namespace fl {
namespace server {
void ModelStore::Initialize(const std::vector<InputWeight> &feature_map, uint32_t max_count) {
  auto latest_iteration_num = cache::InstanceContext::Instance().iteration_num() - 1;
  MS_LOG(INFO) << "Latest iteration num is " << latest_iteration_num;
  max_model_count_ = max_count;
  InitModel(feature_map);
  MS_EXCEPTION_IF_NULL(initial_model_);
  if (!LocalMetaStore::GetInstance().verifyAggregationFeatureMap(initial_model_)) {
    MS_LOG(EXCEPTION) << "Verify feature map failed for initial model.";
  }
  iteration_to_model_[latest_iteration_num] = initial_model_;
  for (const auto &item : mindspore::fl::compression::kCompressTypeMap) {
    iteration_to_compress_model_[latest_iteration_num][item.first] =
      AssignNewCompressModelMemory(item.first, initial_model_);
  }
  model_size_ = initial_model_->model_size;
  MS_LOG(INFO) << "Model store checkpoint dir is: " << FLContext::instance()->checkpoint_dir();
}

void ModelStore::InitModel(const std::vector<InputWeight> &feature_map) {
  size_t model_size = 0;
  for (auto &feature : feature_map) {
    if (feature.name.empty()) {
      MS_LOG(EXCEPTION) << "Feature name cannot be empty";
    }
    if (feature.data == nullptr) {
      MS_LOG(EXCEPTION) << "Feature data cannot be nullptr";
    }
    if (feature.size <= 0 || feature.size >= UINT32_MAX) {
      MS_LOG(EXCEPTION) << "Feature size " << feature.size << "  cannot <=0 or >=UINT32_MAX";
    }
    model_size += feature.size;
  }
  if (model_size == 0 || model_size >= INT32_MAX) {
    MS_LOG(EXCEPTION) << "Model size " << model_size << " cannot <=0 or >=UINT32_MAX";
  }
  // Assign new memory for the model.
  initial_model_ = AllocNewModelItem(model_size);
  initial_model_->model_size = model_size;
  initial_model_->weight_data.resize(model_size);
  auto model_data = initial_model_->weight_data.data();
  size_t cur_offset = 0;
  for (auto &feature : feature_map) {
    auto ret = memcpy_s(model_data + cur_offset, feature.size, feature.data, feature.size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s failed, ret " << ret << ", feature size: " << feature.size
                        << ", cur offset: " << cur_offset;
    }
    auto &weight_item = initial_model_->weight_items[feature.name];
    weight_item.name = feature.name;
    weight_item.offset = cur_offset;
    weight_item.size = feature.size;
    weight_item.shape = feature.shape;
    weight_item.type = feature.type;
    weight_item.require_aggr = feature.require_aggr;
    cur_offset += feature.size;

    MS_LOG(INFO) << "Aggregate Weight full name is " << weight_item.name << ", weight byte size is "
                 << weight_item.size;
  }
  LocalMetaStore::GetInstance().put_aggregation_feature_map(initial_model_);
}

bool ModelStore::StoreModelByIterNum(size_t iteration, const void *proto_model_data, size_t len) {
  if (proto_model_data == nullptr) {
    MS_LOG(WARNING) << "proto_model_data is nullptr.";
    return false;
  }
  ProtoModel proto_model;
  if (!proto_model.ParseFromArray(proto_model_data, static_cast<int>(len))) {
    MS_LOG_WARNING << "Failed to parse data to ProtoModel object";
    return false;
  }
  if (proto_model.iteration_num() != iteration) {
    MS_LOG_WARNING << "The iteration num " << proto_model.iteration_num() << " in ProtoModel != expect iteration num "
                   << iteration;
    return false;
  }
  std::map<std::string, Address> new_model;
  for (auto &proto_feature : proto_model.weights()) {
    auto name = proto_feature.name();
    Address address;
    address.addr = const_cast<char *>(proto_feature.data().data());
    address.size = proto_feature.data().size();
    new_model[name] = address;
  }
  return StoreModelByIterNum(iteration, new_model);
}

bool ModelStore::StoreModelByIterNum(size_t iteration, const std::map<std::string, Address> &new_model) {
  if (new_model.empty()) {
    MS_LOG(WARNING) << "Model feature_map is empty.";
    return false;
  }
  std::unique_lock<std::mutex> lock(model_mtx_);
  // Erase all the model whose iteration >= the iteration of model to be saved.
  // Ensure the saved model is the latest model.
  for (auto it = iteration_to_model_.begin(); it != iteration_to_model_.end();) {
    if (it->first >= iteration) {
      it = iteration_to_model_.erase(it);
    } else {
      ++it;
    }
  }
  if (iteration_to_model_.size() >= max_model_count_) {
    (void)iteration_to_model_.erase(iteration_to_model_.begin());
  }
  // Copy new model data to the stored model.
  auto stored_model = AssignNewModelMemory();
  if (stored_model == nullptr) {
    MS_LOG_ERROR << "Failed to AssignNewModelMemory";
    return false;
  }
  const auto &weight_items = stored_model->weight_items;
  auto weight_data_base = stored_model->weight_data.data();
  for (const auto &weight : new_model) {
    const std::string &weight_name = weight.first;
    auto it = weight_items.find(weight_name);
    if (it == weight_items.end()) {
      MS_LOG(WARNING) << "The stored model has no weight " << weight_name;
      return false;
    }
    auto &weight_item = it->second;
    if (!weight_item.require_aggr) {
      continue;
    }
    auto src_addr = weight.second.addr;
    MS_ERROR_IF_NULL_W_RET_VAL(weight.second.addr, false);
    size_t src_size = weight.second.size;
    if (weight_item.size != src_size) {
      MS_LOG(WARNING) << "The weight size to store " << src_size << " != the expected weight size " << weight_item.size
                      << ", weight: " << weight_name;
      return false;
    }
    int ret = memcpy_s(weight_data_base + weight_item.offset, weight_item.size, src_addr, src_size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }
  }
  iteration_to_model_[iteration] = stored_model;
  OnIterationUpdate();
  return true;
}

bool ModelStore::StoreModelByIterNum(size_t iteration, const ModelItemPtr &new_model_ptr) {
  if (new_model_ptr == nullptr || new_model_ptr->weight_data.empty() || new_model_ptr->weight_items.empty()) {
    MS_LOG(WARNING) << "Model cannot be empty.";
    return false;
  }
  std::unique_lock<std::mutex> lock(model_mtx_);
  // Erase all the model whose iteration >= the iteration of model to be saved.
  // Ensure the saved model is the latest model.
  for (auto it = iteration_to_model_.begin(); it != iteration_to_model_.end();) {
    if (it->first >= iteration) {
      it = iteration_to_model_.erase(it);
    } else {
      ++it;
    }
  }
  if (iteration_to_model_.size() >= max_model_count_) {
    (void)iteration_to_model_.erase(iteration_to_model_.begin());
  }
  iteration_to_model_[iteration] = new_model_ptr;
  OnIterationUpdate();
  return true;
}

ModelItemPtr ModelStore::GetModelByIterNum(size_t iteration) {
  std::unique_lock<std::mutex> lock(model_mtx_);
  auto model_it = iteration_to_model_.find(iteration);
  if (model_it == iteration_to_model_.end()) {
    return nullptr;
  }
  return model_it->second;
}

std::pair<size_t, ModelItemPtr> ModelStore::GetLatestModel() {
  std::unique_lock<std::mutex> lock(model_mtx_);
  if (iteration_to_model_.empty()) {
    return {};
  }
  auto latest_model = iteration_to_model_.rbegin();
  return {latest_model->first, latest_model->second};
}

std::map<std::string, AddressPtr> ModelStore::GetCompressModelByIterNum(size_t iteration,
                                                                        schema::CompressType compressType) {
  std::unique_lock<std::mutex> lock(model_mtx_);
  std::map<std::string, AddressPtr> compressModel = {};
  if (iteration_to_compress_model_.count(iteration) == 0) {
    lock.unlock();
    auto no_compress_model = GetModelByIterNum(iteration);
    if (no_compress_model == nullptr) {
      MS_LOG(ERROR) << "Compress Model for iteration " << iteration << " is not stored.";
      return compressModel;
    }
    StoreCompressModelByIterNum(iteration, no_compress_model);
    lock.lock();
  }
  std::map<schema::CompressType, std::shared_ptr<MemoryRegister>> compress_model_map =
    iteration_to_compress_model_[iteration];
  if (compress_model_map.count(compressType) == 0) {
    MS_LOG(ERROR) << "Compress Model for compress type " << compressType << " is not stored.";
    return compressModel;
  }
  compressModel = iteration_to_compress_model_[iteration][compressType]->addresses();
  return compressModel;
}

void ModelStore::Reset() {
  std::unique_lock<std::mutex> lock(model_mtx_);
  initial_model_ = iteration_to_model_.rbegin()->second;
  iteration_to_model_.clear();
  iteration_to_model_[kInitIterationNum] = initial_model_;
  OnIterationUpdate();
}

const std::map<size_t, ModelItemPtr> &ModelStore::iteration_to_model() {
  std::unique_lock<std::mutex> lock(model_mtx_);
  return iteration_to_model_;
}

const std::map<size_t, CompressTypeMap> &ModelStore::iteration_to_compress_model() {
  std::unique_lock<std::mutex> lock(model_mtx_);
  return iteration_to_compress_model_;
}

size_t ModelStore::model_size() const { return model_size_; }

ModelItemPtr ModelStore::AllocNewModelItem(size_t model_size) {
  std::unique_lock<std::mutex> lock(model_cache_mtx_);
  auto &cache_list = empty_model_cache_[model_size];
  for (auto &item : cache_list) {
    if (item.use_count() == 1) {
      return item;
    }
  }
  auto model = std::make_shared<ModelItem>();
  if (model == nullptr) {
    return nullptr;
  }
  model->weight_data.resize(model_size);
  model->model_size = model_size;
  constexpr size_t max_cache_size = 6;
  if (cache_list.size() < max_cache_size) {
    cache_list.push_back(model);
  }
  return model;
}

ModelItemPtr ModelStore::AssignNewModelMemory() {
  if (initial_model_ == nullptr || initial_model_->weight_data.empty() || initial_model_->weight_items.empty()) {
    MS_LOG(WARNING) << "Load model is invalid.";
    return nullptr;
  }
  // Assign new memory for the model.
  auto new_model = AllocNewModelItem(initial_model_->weight_data.size());
  MS_ERROR_IF_NULL_W_RET_VAL(new_model, nullptr);
  auto &weight_data = new_model->weight_data;
  if (weight_data.size() != initial_model_->weight_data.size()) {
    return nullptr;
  }
  auto ret = memset_s(weight_data.data(), weight_data.size(), 0, weight_data.size());
  if (ret != EOK) {
    MS_LOG_WARNING << "Failed to init weight data, memset_s return " << ret;
    return nullptr;
  }
  new_model->weight_items = initial_model_->weight_items;
  auto new_weight_base = weight_data.data();
  auto src_weight_base = initial_model_->weight_data.data();
  for (auto &weight : initial_model_->weight_items) {
    auto &weight_info = weight.second;
    if (!weight_info.require_aggr) {
      auto dst_size = weight_data.size() - weight_info.offset;
      ret = memcpy_s(new_weight_base + weight_info.offset, dst_size, src_weight_base + weight_info.offset,
                     weight_info.size);
      if (ret != EOK) {
        MS_LOG_WARNING << "Failed to init weight data, memcpy_s return " << ret << ", offset: " << weight_info.offset
                       << ", weight size: " << weight_info.size << ", model size: " << weight_data.size();
        return nullptr;
      }
    }
  }
  return new_model;
}

std::shared_ptr<MemoryRegister> ModelStore::AssignNewCompressModelMemory(schema::CompressType compressType,
                                                                         const ModelItemPtr &model) {
  if (model == nullptr || model->weight_items.empty() || model->weight_data.empty()) {
    MS_LOG(EXCEPTION) << "Model feature map is empty.";
    return nullptr;
  }
  std::map<std::string, std::vector<float>> feature_maps;
  for (auto &feature : model->weight_items) {
    auto weight_fullname = feature.first;
    auto weight_data = reinterpret_cast<float *>(model->weight_data.data() + feature.second.offset);
    std::vector<float> weight_data_vector{weight_data, weight_data + feature.second.size / sizeof(float)};
    feature_maps[weight_fullname] = weight_data_vector;
  }

  std::map<std::string, mindspore::fl::compression::CompressWeight> compressWeights;
  bool status = mindspore::fl::compression::CompressExecutor::GetInstance().construct_compress_weight(
    &compressWeights, feature_maps, compressType);
  if (!status) {
    MS_LOG(ERROR) << "Encode failed!";
    return nullptr;
  }

  // Assign new memory for the compress model.
  std::shared_ptr<MemoryRegister> memory_register = std::make_shared<MemoryRegister>();
  MS_ERROR_IF_NULL_W_RET_VAL(memory_register, nullptr);
  MS_LOG(INFO) << "Register compressWeight for compressType: " << schema::EnumNameCompressType(compressType);

  for (const auto &compressWeight : compressWeights) {
    if (compressType == schema::CompressType_QUANT) {
      std::string compress_weight_name = compressWeight.first;
      std::string min_val_name = compress_weight_name + "." + kMinVal;
      std::string max_val_name = compress_weight_name + "." + kMaxVal;
      size_t compress_weight_size = compressWeight.second.compress_data_len * sizeof(int8_t);
      auto compress_weight_data = std::make_unique<char[]>(compress_weight_size);
      auto src_data_size = compress_weight_size;
      auto dst_data_size = compress_weight_size;
      int ret =
        memcpy_s(compress_weight_data.get(), dst_data_size, compressWeight.second.compress_data.data(), src_data_size);
      if (ret != 0) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return nullptr;
      }
      memory_register->RegisterArray(compress_weight_name, &compress_weight_data, compress_weight_size);
      size_t float_size = 1;
      auto min_val_ptr = std::make_unique<float>(compressWeight.second.min_val);
      auto max_val_ptr = std::make_unique<float>(compressWeight.second.max_val);

      memory_register->RegisterParameter(min_val_name, &min_val_ptr, float_size);
      memory_register->RegisterParameter(max_val_name, &max_val_ptr, float_size);
    }
  }
  return memory_register;
}

void ModelStore::StoreCompressModelByIterNum(size_t iteration, const ModelItemPtr &new_model) {
  std::unique_lock<std::mutex> lock(model_mtx_);
  if (iteration_to_compress_model_.count(iteration) != 0) {
    MS_LOG(WARNING) << "Compress Model for iteration " << iteration << " is already stored";
    return;
  }
  if (new_model == nullptr || new_model->weight_items.empty() || new_model->weight_data.empty()) {
    MS_LOG(ERROR) << "Compress Model feature map is empty.";
    return;
  }

  iteration_to_compress_model_[iteration] = {};
  if (iteration_to_compress_model_.size() >= max_model_count_) {
    auto compress_model_map = iteration_to_compress_model_.begin()->second;
    compress_model_map.clear();
    (void)iteration_to_compress_model_.erase(iteration_to_compress_model_.begin());
  }

  for (const auto &item : mindspore::fl::compression::kCompressTypeMap) {
    auto memory_register = AssignNewCompressModelMemory(item.first, new_model);
    MS_ERROR_IF_NULL_WO_RET_VAL(memory_register);
    iteration_to_compress_model_[iteration][item.first] = memory_register;
  }
}

void ModelStore::RelModelResponseCache(const void *data, size_t datalen, void *extra) {
  MS_ERROR_IF_NULL_WO_RET_VAL(data);
  auto &instance = GetInstance();
  std::unique_lock<std::mutex> lock(instance.model_response_cache_lock_);
  auto it =
    std::find_if(instance.model_response_cache_.begin(), instance.model_response_cache_.end(),
                 [data](const HttpResponseModelCache &item) { return item.cache && item.cache->data() == data; });
  if (it == instance.model_response_cache_.end()) {
    MS_LOG(WARNING) << "Model response cache has been releaed";
    return;
  }
  if (it->reference_count > 0) {
    it->reference_count -= 1;
    instance.total_sub_reference_count++;
  }
}

VectorPtr ModelStore::GetModelResponseCache(const std::string &round_name, size_t cur_iteration_num,
                                            size_t model_iteration_num, const std::string &compress_type) {
  std::unique_lock<std::mutex> lock(model_response_cache_lock_);
  auto it = std::find_if(
    model_response_cache_.begin(), model_response_cache_.end(),
    [&round_name, cur_iteration_num, model_iteration_num, &compress_type](const HttpResponseModelCache &item) {
      return item.round_name == round_name && item.cur_iteration_num == cur_iteration_num &&
             item.model_iteration_num == model_iteration_num && item.compress_type == compress_type;
    });
  if (it == model_response_cache_.end()) {
    return nullptr;
  }
  it->reference_count += 1;
  total_add_reference_count += 1;
  return it->cache;
}

VectorPtr ModelStore::StoreModelResponseCache(const std::string &round_name, size_t cur_iteration_num,
                                              size_t model_iteration_num, const std::string &compress_type,
                                              const void *data, size_t datalen) {
  std::unique_lock<std::mutex> lock(model_response_cache_lock_);
  auto it = std::find_if(
    model_response_cache_.begin(), model_response_cache_.end(),
    [&round_name, cur_iteration_num, model_iteration_num, &compress_type](const HttpResponseModelCache &item) {
      return item.round_name == round_name && item.cur_iteration_num == cur_iteration_num &&
             item.model_iteration_num == model_iteration_num && item.compress_type == compress_type;
    });
  if (it != model_response_cache_.end()) {
    it->reference_count += 1;
    total_add_reference_count += 1;
    return it->cache;
  }
  auto cache = std::make_shared<std::vector<uint8_t>>(datalen);
  if (cache == nullptr) {
    MS_LOG(ERROR) << "Malloc data of size " << datalen << " failed";
    return nullptr;
  }
  auto ret = memcpy_s(cache->data(), cache->size(), data, datalen);
  if (ret != 0) {
    MS_LOG(ERROR) << "memcpy_s  error, errorno(" << ret << ")";
    return nullptr;
  }
  HttpResponseModelCache item;
  item.round_name = round_name;
  item.cur_iteration_num = cur_iteration_num;
  item.model_iteration_num = model_iteration_num;
  item.compress_type = compress_type;
  item.cache = cache;
  item.reference_count = 1;
  total_add_reference_count += 1;
  model_response_cache_.push_back(item);
  return cache;
}

void ModelStore::OnIterationUpdate() {
  std::unique_lock<std::mutex> lock(model_response_cache_lock_);
  for (auto it = model_response_cache_.begin(); it != model_response_cache_.end();) {
    if (it->reference_count == 0) {
      it->cache = nullptr;
      it = model_response_cache_.erase(it);
    } else {
      ++it;
    }
  }
  MS_LOG(INFO) << "Current model cache number: " << model_response_cache_.size()
               << ", total add and sub reference count: " << total_add_reference_count << ", "
               << total_sub_reference_count;
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
