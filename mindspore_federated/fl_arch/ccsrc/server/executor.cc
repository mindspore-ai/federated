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

#include "server/executor.h"
#include <set>
#include <memory>
#include <string>
#include <vector>
#include "distributed_cache/instance_context.h"
#include "distributed_cache/server.h"
#include "distributed_cache/counter.h"
#include "server/model_store.h"
#include "server/server.h"
#include "server/kernel/fed_avg_kernel.h"

namespace mindspore {
namespace fl {
namespace server {
void Executor::Initialize(const std::vector<InputWeight> &feature_map, const std::shared_ptr<ServerNode> &server_node) {
  ModelStore::GetInstance().Initialize(feature_map);
  if (!ResetAggregationStatus()) {
    MS_LOG_EXCEPTION << "Failed to reset aggregation status";
  }
  server_node_ = server_node;
  initialized_ = true;
}

bool Executor::initialized() const { return initialized_; }

FlStatus Executor::CheckUpdatedModel(const std::map<std::string, Address> &feature_map,
                                     const std::string &update_model_fl_id) {
  std::unique_lock<std::mutex> lock(parameter_mutex_);
  for (auto &param_item : param_aggregation_info_) {
    auto &param_name = param_item.first;
    auto &param_aggr = param_item.second;
    if (!param_aggr.requires_aggr) {
      continue;
    }
    auto it = feature_map.find(param_name);
    if (it == feature_map.end()) {
      auto reason = "The updated weight of parameter " + param_name + " is missing, fl id: " + update_model_fl_id;
      MS_LOG_WARNING << reason;
      return {kFlFailed, reason};
    }
    auto upload_data = it->second;
    if (param_aggr.weight_size != upload_data.size) {
      MS_LOG_WARNING << "The weight bytes size " << upload_data.size << " uploaded of parameter " << param_name
                     << " != expected size " << param_aggr.weight_size << ", fl id: " << update_model_fl_id;
      auto reason = "Updating weight " + param_name + " failed, fl id: " + update_model_fl_id;
      return {kFlFailed, reason};
    }
  }
  return kFlSuccess;
}

void Executor::HandleModelUpdate(const std::map<std::string, Address> &feature_map, size_t data_size) {
  std::unique_lock<std::mutex> lock(parameter_mutex_);
  for (auto &param_item : param_aggregation_info_) {
    auto &param_name = param_item.first;
    auto &param_aggr = param_item.second;
    if (!param_aggr.requires_aggr) {
      continue;
    }
    auto it = feature_map.find(param_name);
    if (it == feature_map.end()) {
      continue;
    }
    auto upload_data = it->second;

    MS_LOG(DEBUG) << "Do UpdateModel for parameter " << param_name;
    kernel::FedAvgKernel<float, size_t>::Launch(upload_data, data_size, &param_aggr);
  }
}

bool Executor::OnReceiveModelWeight(const uint8_t *proto_model_data, size_t len) {
  ProtoModel proto_model;
  if (!proto_model.ParseFromArray(proto_model_data, static_cast<int>(len))) {
    MS_LOG_WARNING << "Failed to parse data to ProtoModel object";
    return false;
  }
  auto iteration_num = cache::InstanceContext::Instance().iteration_num();
  if (proto_model.iteration_num() != iteration_num) {
    MS_LOG_WARNING << "The iteration num " << proto_model.iteration_num() << " in ProtoModel != iteration num "
                   << iteration_num << " of local";
    return false;
  }
  std::unique_lock<std::mutex> lock(parameter_mutex_);
  for (const auto &param : proto_model.weights()) {
    const std::string &param_name = param.name();
    if (param_aggregation_info_.count(param_name) == 0) {
      MS_LOG(WARNING) << "Weight " << param_name << " is not registered in server.";
      continue;
    }
    auto &param_aggr = param_aggregation_info_[param_name];
    int ret = memcpy_s(param_aggr.weight_data, param_aggr.weight_size, param.data().data(), param.data().size());
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << "), src size: " << param.data().size()
                    << ", dst size: " << param_aggr.weight_size;
      return false;
    }
  }
  SetIterationModelFinished();
  return true;
}

void Executor::SetIterationModelFinished() { model_finished_ = true; }

bool Executor::IsIterationModelFinished(uint64_t iteration_num) const {
  auto cur_iter = cache::InstanceContext::Instance().iteration_num();
  if (cur_iter != iteration_num) {
    return false;
  }
  return model_finished_;
}

std::map<std::string, Address> Executor::ParseFeatureMap(const schema::RequestPushWeight *push_weight_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(push_weight_req, {});
  std::map<std::string, Address> upload_feature_map;
  auto fbs_feature_map = push_weight_req->feature_map();
  MS_ERROR_IF_NULL_W_RET_VAL(fbs_feature_map, upload_feature_map);
  for (uint32_t i = 0; i < fbs_feature_map->size(); i++) {
    auto feature = fbs_feature_map->Get(i);
    if (feature == nullptr || feature->weight_fullname() == nullptr || feature->data() == nullptr) {
      MS_LOG_WARNING << "Feature parsed from flatbuffer is invalid";
      return {};
    }
    std::string weight_full_name = feature->weight_fullname()->str();
    float *weight_data = const_cast<float *>(feature->data()->data());
    size_t weight_size = feature->data()->size() * sizeof(float);
    upload_feature_map[weight_full_name] = {weight_data, weight_size};
  }
  return upload_feature_map;
}

bool Executor::HandlePushWeight(const std::map<std::string, Address> &feature_map) {
  std::unique_lock<std::mutex> lock(parameter_mutex_);
  for (const auto &trainable_param : feature_map) {
    const std::string &param_name = trainable_param.first;
    if (param_aggregation_info_.count(param_name) == 0) {
      MS_LOG(WARNING) << "Weight " << param_name << " is not registered in server.";
      continue;
    }
    auto &param_aggr = param_aggregation_info_[param_name];
    const Address &new_weight = trainable_param.second;
    MS_ERROR_IF_NULL_W_RET_VAL(param_aggr.weight_data, false);
    MS_ERROR_IF_NULL_W_RET_VAL(new_weight.addr, false);
    int ret = memcpy_s(param_aggr.weight_data, param_aggr.weight_size, new_weight.addr, new_weight.size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }
  }
  SetIterationModelFinished();
  std::map<std::string, std::string> broadcast_src_nodes = {
    {cache::Server::Instance().node_id(), cache::Server::Instance().tcp_address()}};
  BroadcastModelWeight(broadcast_src_nodes);
  return true;
}

FlStatus Executor::BuildPullWeightRsp(size_t iteration, const std::vector<std::string> &param_names, FBBuilder *fbb) {
  std::string reason;
  if (fbb == nullptr) {
    reason = "fbb is nullptr.";
    MS_LOG(ERROR) << "fbb is nullptr.";
    return {kFlFailed, reason};
  }
  std::unique_lock<std::mutex> lock(parameter_mutex_);
  std::vector<flatbuffers::Offset<schema::FeatureMap>> fbs_feature_maps;
  for (const auto &param_name : param_names) {
    if (param_aggregation_info_.count(param_name) == 0) {
      reason = "Parameter " + param_name + " is not registered in server.";
      MS_LOG(ERROR) << reason;
      return {kRequestError, reason};
    }
    const auto &param_aggr = param_aggregation_info_[param_name];
    MS_ERROR_IF_NULL_W_RET_VAL(param_aggr.weight_data, kFlFailed);

    auto fbs_weight_fullname = fbb->CreateString(param_name);
    auto fbs_weight_data =
      fbb->CreateVector(reinterpret_cast<float *>(param_aggr.weight_data), param_aggr.weight_size / sizeof(float));
    auto fbs_feature_map = schema::CreateFeatureMap(*fbb, fbs_weight_fullname, fbs_weight_data);
    fbs_feature_maps.push_back(fbs_feature_map);
  }
  reason = "Pulling weight by weight names for iteration " + std::to_string(iteration) + " success.";
  auto fbs_reason = fbb->CreateString(reason);
  auto fbs_feature_maps_vector = fbb->CreateVector(fbs_feature_maps);

  schema::ResponsePullWeightBuilder rsp_pull_weight_builder(*fbb);
  rsp_pull_weight_builder.add_retcode(SizeToInt(schema::ResponseCode_SUCCEED));
  rsp_pull_weight_builder.add_reason(fbs_reason);
  rsp_pull_weight_builder.add_iteration(SizeToInt(iteration));
  rsp_pull_weight_builder.add_feature_map(fbs_feature_maps_vector);
  auto rsp_pull_weight = rsp_pull_weight_builder.Finish();
  fbb->Finish(rsp_pull_weight);
  return kFlSuccess;
}

FlStatus Executor::HandlePullWeightRequest(const uint8_t *req_data, size_t len, FBBuilder *fbb) {
  std::string reason;
  if (!IsAggregationDone() || !IsUnmasked()) {
    reason = "The aggregation for the weights is not done yet.";
    return FlStatus(kAggregationNotDone, reason);
  }
  if (req_data == nullptr || fbb == nullptr) {
    reason = "System error: Input parameter invalid";
    MS_LOG(WARNING) << reason;
    return FlStatus(kRequestError, reason);
  }
  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::RequestPullWeight>()) {
    reason = "The schema of RequestPushWeight is invalid.";
    MS_LOG(ERROR) << reason;
    return FlStatus(kRequestError, reason);
  }
  const schema::RequestPullWeight *pull_weight_req = flatbuffers::GetRoot<schema::RequestPullWeight>(req_data);
  if (pull_weight_req == nullptr) {
    reason = "Building flatbuffers schema failed for RequestPullWeight";
    MS_LOG(WARNING) << reason;
    return FlStatus(kRequestError, reason);
  }
  std::map<std::string, AddressPtr> feature_maps = {};
  size_t current_iter = cache::InstanceContext::Instance().iteration_num();
  size_t pull_weight_iter = IntToSize(pull_weight_req->iteration());
  // The iteration from worker should be the same as server's, otherwise return SucNotReady so that worker could retry.
  if (pull_weight_iter != current_iter) {
    reason = "PullWeight iteration " + std::to_string(pull_weight_iter) +
             " is invalid. Server current iteration: " + std::to_string(current_iter);
    MS_LOG(WARNING) << reason;
    return FlStatus(kNotReadyError, reason);
  }

  std::vector<std::string> weight_names = {};
  auto weights_names_fbs = pull_weight_req->weight_names();
  if (weights_names_fbs == nullptr) {
    reason = "weights_names_fbs is nullptr.";
    MS_LOG(WARNING) << reason;
    return FlStatus(kRequestError, reason);
  }
  for (uint32_t i = 0; i < weights_names_fbs->size(); i++) {
    weight_names.push_back(weights_names_fbs->Get(i)->str());
  }
  auto status = BuildPullWeightRsp(current_iter, weight_names, fbb);
  if (!status.IsSuccess()) {
    return status;
  }
  MS_LOG(INFO) << "Pulling weight for iteration " << current_iter << " succeeds.";
  return kFlSuccess;
}

void Executor::SetSkipAggregation() { is_aggregation_skip_ = true; }

bool Executor::IsAggregationSkip() const { return is_aggregation_skip_; }

bool Executor::IsAggregationDone() const { return is_aggregation_done_; }

bool Executor::GetServersForAllReduce(std::map<std::string, std::string> *all_reduce_server_map) {
  std::map<std::string, std::string> all_server_map;
  std::unordered_map<std::string, uint64_t> count_server_map;
  auto cache_ret = cache::Server::Instance().GetAllServersRealtime(&all_server_map);
  if (!cache_ret.IsSuccess()) {
    MS_LOG_WARNING << "Failed to obtain all servers real-time";
    return false;
  }
  cache_ret = cache::Counter::Instance().GetPerServerCountMap(kUpdateModelKernel, &count_server_map);
  if (!cache_ret.IsSuccess()) {
    MS_LOG_WARNING << "Failed to obtain updateModel count of all servers";
    return false;
  }
  for (auto &item : all_server_map) {
    auto node_id = item.first;
    if (count_server_map.count(node_id) > 0) {
      all_reduce_server_map->emplace(std::make_pair(node_id, item.second));
    }
  }
  if (!cache::Counter::Instance().ReachThreshold(kUpdateModelKernel)) {
    MS_LOG_WARNING << "The update model count on all alive servers is less than the threshold";
    return false;
  }
  return true;
}

void Executor::BroadcastModelWeight(const std::map<std::string, std::string> &broadcast_src_server_map) {
  if (broadcast_src_server_map.empty()) {
    MS_LOG_WARNING << "The broadcast source server map is empty";
    return;
  }
  auto cur_node_id = cache::Server::Instance().node_id();
  // only the first node broadcast weight
  if (broadcast_src_server_map.begin()->first != cur_node_id) {
    return;
  }
  std::map<std::string, std::string> all_server_map;
  auto cache_ret = cache::Server::Instance().GetAllServersRealtime(&all_server_map);
  if (!cache_ret.IsSuccess()) {
    MS_LOG_WARNING << "Failed to obtain all servers";
    return;
  }
  std::map<std::string, std::string> broadcast_server_map;
  for (auto &item : all_server_map) {
    auto node_id = item.first;
    if (broadcast_src_server_map.count(node_id) == 0) {
      broadcast_server_map.emplace(std::make_pair(node_id, item.second));
    }
  }
  if (broadcast_server_map.empty()) {
    return;
  }
  auto curr_iter_num = cache::InstanceContext::Instance().iteration_num();
  auto model = GetModel();
  ProtoModel proto_model;
  TransModel2ProtoModel(curr_iter_num, model, &proto_model);
  auto model_str = proto_model.SerializeAsString();
  server::Server::GetInstance().BroadcastModelWeight(model_str, broadcast_server_map);
}

// Invoked by counter event handle, and runs on the same thread as method RunWeightAggregation.
// A lock is not required.
void Executor::TodoUnmask() {
  can_unmask_ = true;
  if (!is_aggregation_done_) {
    return;
  }
  Unmask();
}

void Executor::OnPushMetrics() {
  if (FLContext::instance()->resetter_round() == ResetterRound::kPushMetrics) {
    std::string reason = "Push metrics finished! This iteration is valid. Proceed to next iteration.";
    FinishIteration(true, reason);
  }
}

// Invoked by counter event handle, and runs on the same thread as method TodoUnmask.
// A lock is not required.
void Executor::RunWeightAggregation() {
  auto curr_iter_num = cache::InstanceContext::Instance().iteration_num();
  std::map<std::string, std::string> server_map;
  auto valid = GetServersForAllReduce(&server_map);
  if (!valid) {
    std::string reason = "Weight aggregation failed, current iteration: " + std::to_string(curr_iter_num);
    MS_LOG(WARNING) << reason;
    FinishIteration(false, reason);
    return;
  }
  auto node_id = cache::Server::Instance().node_id();
  if (server_map.count(node_id) == 0) {
    MS_LOG_INFO << "Skip current node, this node does not contribute the updateModel count";
    Executor::GetInstance().SetSkipAggregation();
    return;
  }
  valid = RunWeightAggregationInner(server_map);
  if (!valid) {
    std::string reason = "Weight aggregation failed, current iteration: " + std::to_string(curr_iter_num);
    MS_LOG(WARNING) << reason;
    FinishIteration(false, reason);
    return;
  }
  size_t total_data_size = LocalMetaStore::GetInstance().value<size_t>(kCtxFedAvgTotalDataSize);
  MS_LOG(INFO) << "Total data size for iteration " << curr_iter_num << " is " << total_data_size;
  if (FLContext::instance()->resetter_round() == ResetterRound::kUpdateModel) {
    SetIterationModelFinished();
    BroadcastModelWeight(all_reduce_server_map_);
    std::string reason = "Weight aggregation finished! This iteration is valid. Proceed to next iteration.";
    FinishIteration(true, reason);
    return;
  }
  if (!can_unmask_) {
    return;
  }
  Unmask();
}

bool Executor::RunWeightAggregationInner(const std::map<std::string, std::string> &server_map) {
  all_reduce_server_map_ = server_map;
  if (server_map.size() == 1) {
    MS_LOG_INFO << "Servers count for RunWeightAggregation is 1";
  }
  std::unique_lock<std::mutex> lock(parameter_mutex_);
  for (auto &item : param_aggregation_info_) {
    auto &param_aggr = item.second;
    if (!param_aggr.requires_aggr) {
      continue;
    }
    if (!kernel::FedAvgKernel<float, size_t>::AllReduce(server_map, &param_aggr)) {
      MS_LOG(WARNING) << "Failed to run aggregation for " << param_aggr.name;
      return false;
    }
  }
  is_aggregation_done_ = true;
  return true;
}

void Executor::FinishIteration(bool is_last_iter_valid, const std::string &in_reason) {
  cache::InstanceContext::Instance().NotifyNext(is_last_iter_valid, in_reason);
}

bool Executor::ResetAggregationStatus() {
  is_aggregation_done_ = false;
  is_aggregation_skip_ = false;
  unmasked_ = false;
  can_unmask_ = false;
  all_reduce_server_map_.clear();
  std::unique_lock<std::mutex> lock(parameter_mutex_);
  model_finished_ = false;
  model_aggregation_ = ModelStore::GetInstance().AssignNewModelMemory();
  if (model_aggregation_ == nullptr) {
    MS_LOG_ERROR << "Failed to alloc new model";
    return false;
  }
  param_aggregation_info_.clear();
  auto weight_data = model_aggregation_->weight_data.data();
  for (auto &item : model_aggregation_->weight_items) {
    auto &weight_item = item.second;
    ParamAggregationInfo info;
    info.name = weight_item.name;
    info.weight_data = weight_data + weight_item.offset;
    info.weight_size = weight_item.size;
    info.data_size = 0;
    info.requires_aggr = weight_item.requires_aggr;
    param_aggregation_info_[info.name] = info;
  }
  return true;
}

ModelItemPtr Executor::GetModel() { return model_aggregation_; }

ModelItemPtr Executor::GetModelByIteration(uint64_t iteration_num) {
  // step0: all reduce, unmask -> FinishIteration(true) -> GetModel valid
  // step1: notify next or HandlePushWeight: SetIterationModelFinished
  // step2: save model in ModelStore
  // step3: reset weight in GetModel, model_finish=false
  ModelItemPtr model_ret = model_aggregation_;  // invalid when >= step3
  if (IsIterationModelFinished(iteration_num)) {
    return model_ret;
  }
  // valid when >= step2
  return ModelStore::GetInstance().GetModelByIterNum(iteration_num);
}

bool Executor::GetModelByIteration(uint64_t iteration_num, ProtoModel *proto_model) {
  auto model = GetModelByIteration(iteration_num);
  if (model == nullptr) {
    return false;
  }
  return TransModel2ProtoModel(iteration_num, model, proto_model);
}

bool Executor::TransModel2ProtoModel(uint64_t iteration_num, const ModelItemPtr &model, ProtoModel *proto_model) {
  if (model == nullptr || proto_model == nullptr) {
    return false;
  }
  proto_model->set_fl_name(cache::InstanceContext::Instance().fl_name());
  proto_model->set_instance_name(cache::InstanceContext::Instance().instance_name());
  auto instance_state = cache::InstanceContext::Instance().instance_state();
  InstanceState proto_state = InstanceState::kStateRunning;
  if (instance_state == cache::InstanceState::kStateDisable) {
    proto_state = InstanceState::kStateDisable;
  } else if (instance_state == cache::InstanceState::kStateFinish) {
    proto_state = InstanceState::kStateFinish;
  } else if (instance_state == cache::InstanceState::kStateStop) {
    proto_state = InstanceState::kStateStop;
  }
  proto_model->set_instance_state(proto_state);
  proto_model->set_iteration_num(iteration_num);
  auto weight_base = model->weight_data.data();
  for (auto &weight_item : model->weight_items) {
    auto &weight = weight_item.second;
    auto proto_weight = proto_model->add_weights();
    proto_weight->set_name(weight.name);
    proto_weight->set_type(weight.type);
    proto_weight->set_requires_aggr(weight.requires_aggr);
    for (auto &dim : weight.shape) {
      proto_weight->add_shape(static_cast<int64_t>(dim));
    }
    proto_weight->set_data(weight_base + weight.offset, weight.size);
  }
  return true;
}

FlStatus Executor::SyncLatestModelFromOtherServers() {
  if (server_node_ == nullptr) {
    auto reason = "server_node_ cannot be nullptr";
    MS_LOG_ERROR << reason;
    return {kFlFailed, reason};
  }
  // new_iteration_num is the iteration to be updated
  auto updated_iteration = cache::InstanceContext::Instance().new_iteration_num();
  if (updated_iteration <= 0) {
    auto reason = "Invalid iteration number: " + std::to_string(updated_iteration);
    return {kFlFailed, reason};
  }
  auto model_latest_iteration = updated_iteration - 1;
  VectorPtr output = nullptr;
  if (server_node_->GetModelWeight(model_latest_iteration, &output)) {
    auto ret = ModelStore::GetInstance().StoreModelByIterNum(model_latest_iteration, output->data(), output->size());
    if (!ret) {
      auto reason = "Failed to store model synced from other servers";
      return {kFlFailed, reason};
    }
    MS_LOG_INFO << "Sync model success: The model synced from other servers is used as the model of iteration "
                << model_latest_iteration;
  } else {
    auto model = ModelStore::GetInstance().GetLatestModel();
    if (model.second == nullptr) {
      auto reason = "Failed to get latest model from model store";
      return {kFlFailed, reason};
    }
    if (model.first != model_latest_iteration) {
      (void)ModelStore::GetInstance().StoreModelByIterNum(model_latest_iteration, model.second);
    }
    MS_LOG_INFO << "Sync model success: The local model of iteration " << model.first
                << " is used as the model of iteration " << model_latest_iteration;
  }
  return kFlSuccess;
}

void Executor::Unmask() {
  if (IsUnmasked()) {
    return;
  }
  auto curr_iter_num = cache::InstanceContext::Instance().iteration_num();
  auto model = GetModel();
  if (model == nullptr) {
    std::string reason = "Failed to GetModel, current iteration: " + std::to_string(curr_iter_num);
    MS_LOG_WARNING << reason;
    FinishIteration(false, reason);
    return;
  }
  MS_LOG(INFO) << "start unmask";
  auto ret = cipher_unmask_.UnMask(model);
  MS_LOG(INFO) << "end unmask";
  if (!ret) {
    std::string reason = "Failed to unmask, current iteration: " + std::to_string(curr_iter_num);
    MS_LOG_WARNING << reason;
    FinishIteration(false, reason);
    return;
  }
  if (FLContext::instance()->resetter_round() == ResetterRound::kReconstructSeccrets) {
    SetIterationModelFinished();
    BroadcastModelWeight(all_reduce_server_map_);
    std::string reason = "Weight unmask finished! This iteration is valid. Proceed to next iteration.";
    FinishIteration(true, reason);
  }
  unmasked_ = true;
}

bool Executor::IsUnmasked() const {
  std::string encrypt_type = FLContext::instance()->encrypt_type();
  if (encrypt_type == kPWEncryptType) {
    return unmasked_.load();
  } else {
    // If the algorithm of mind armour is not enabled, consider unmasked_ flag as true.
    return true;
  }
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
