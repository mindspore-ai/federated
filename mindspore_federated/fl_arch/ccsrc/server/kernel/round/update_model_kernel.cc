/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "server/kernel/round/update_model_kernel.h"

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "distributed_cache/server.h"
#include "server/server.h"
#include "distributed_cache/timer.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
namespace {
const size_t kLevelNum = 2;
const uint64_t kMaxLevelNum = 2880;
const uint64_t kMinLevelNum = 0;
const int kBase = 10;
const uint64_t kMinuteToSecond = 60;
const uint64_t kSecondToMills = 1000;
const uint64_t kDefaultLevel1 = 5;
const uint64_t kDefaultLevel2 = 15;
}  // namespace
const char *kCountForAggregation = "count_for_aggregation";

void UpdateModelKernel::InitKernel(size_t threshold_count) {
  InitClientVisitedNum();
  InitClientUploadLoss();
  InitClientUploadAccuracy();
  InitEvalDataSize();
  InitTrainDataSize();

  LocalMetaStore::GetInstance().put_value(kCtxUpdateModelThld, threshold_count);
  LocalMetaStore::GetInstance().put_value(kCtxFedAvgTotalDataSize, kInitialDataSizeSum);

  auto first_count_handler = [this]() {};
  auto last_count_handler = [this]() {
    if (!LocalMetaStore::GetInstance().verifyAggregationFeatureMap(Executor::GetInstance().GetModel())) {
      MS_LOG(WARNING) << "Verify feature map failed before aggregation";
    }
    Executor::GetInstance().RunWeightAggregation();
    if (!LocalMetaStore::GetInstance().verifyAggregationFeatureMap(Executor::GetInstance().GetModel())) {
      MS_LOG(WARNING) << "Verify feature map failed after aggregation";
    }
  };
  cache::Counter::Instance().RegisterPerServerCounter(kCountForAggregation, threshold_count, first_count_handler,
                                                      last_count_handler);
  std::string participation_time_level_str = FLContext::instance()->participation_time_level();
  CheckAndTransPara(participation_time_level_str);
}

bool UpdateModelKernel::VerifyUpdateModelRequest(const schema::RequestUpdateModel *update_model_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, false);
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req->fl_id(), false);
  std::string fl_id = update_model_req->fl_id()->str();
  schema::CompressType upload_compress_type = update_model_req->upload_compress_type();

  if (upload_compress_type == schema::CompressType_NO_COMPRESS) {
    auto fbs_feature_map = update_model_req->feature_map();
    MS_ERROR_IF_NULL_W_RET_VAL(fbs_feature_map, false);
    for (uint32_t i = 0; i < fbs_feature_map->size(); i++) {
      auto feature = fbs_feature_map->Get(i);
      if (feature == nullptr || feature->weight_fullname() == nullptr || feature->data() == nullptr) {
        return false;
      }
    }
  } else {
    auto compress_feature_map = update_model_req->compress_feature_map();
    MS_ERROR_IF_NULL_W_RET_VAL(compress_feature_map, false);
    for (uint32_t i = 0; i < compress_feature_map->size(); i++) {
      auto feature = compress_feature_map->Get(i);
      if (feature == nullptr || feature->weight_fullname() == nullptr || feature->compress_data() == nullptr ||
          feature->compress_data()->data() == nullptr) {
        return false;
      }
    }
  }
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req->timestamp(), false);
  float upload_loss = update_model_req->upload_loss();
  if (std::isnan(upload_loss) || std::isinf(upload_loss)) {
    MS_LOG(WARNING) << "The upload loss is nan or inf, client fl id is " << fl_id;
    return false;
  }
  float upload_accuracy = update_model_req->upload_accuracy();
  if (std::isnan(upload_accuracy) || std::isinf(upload_accuracy)) {
    MS_LOG(WARNING) << "The upload accuracy is nan or inf, client fl id is " << fl_id;
    return false;
  }
  return true;
}

bool UpdateModelKernel::Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) {
  MS_LOG(DEBUG) << "Launching UpdateModelKernel kernel.";

  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(WARNING) << reason;
    SendResponseMsg(message, reason.c_str(), reason.size());
    return false;
  }

  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::RequestUpdateModel>()) {
    std::string reason = "The schema of RequestUpdateModel is invalid.";
    BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
    MS_LOG(WARNING) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  const schema::RequestUpdateModel *update_model_req = flatbuffers::GetRoot<schema::RequestUpdateModel>(req_data);
  if (!VerifyUpdateModelRequest(update_model_req)) {
    std::string reason = "Verify flatbuffers schema failed for RequestUpdateModel.";
    BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
    MS_LOG(WARNING) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  ResultCode result_code = ReachThresholdForUpdateModel(fbb, update_model_req);
  if (result_code != ResultCode::kSuccess) {
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  DeviceMeta device_meta;
  result_code = VerifyUpdateModel(update_model_req, fbb, &device_meta);
  if (result_code != ResultCode::kSuccess) {
    MS_LOG(DEBUG) << "Verify updating model failed.";
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }
  std::map<std::string, std::vector<float>> weight_map;
  std::map<std::string, Address> feature_map;
  result_code = ParseAndVerifyFeatureMap(update_model_req, device_meta, fbb, &weight_map, &feature_map);
  if (result_code != ResultCode::kSuccess) {
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    MS_LOG(DEBUG) << "Check model failed.";
    return false;
  }
  result_code = UpdateModel(update_model_req, fbb, device_meta, feature_map);
  if (result_code != ResultCode::kSuccess) {
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    MS_LOG(DEBUG) << "Updating model failed.";
    return false;
  }
  IncreaseAcceptClientNum();
  RecordCompletePeriod(device_meta);
  SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());

  result_code = CountForAggregation();
  if (result_code != ResultCode::kSuccess) {
    return false;
  }
  return true;
}

bool UpdateModelKernel::Reset() {
  MS_LOG(INFO) << "Update model kernel reset!";
  cache::Timer::Instance().StopTimer(name_);
  return true;
}

void UpdateModelKernel::OnLastCountEvent() {}

const std::vector<std::pair<uint64_t, uint32_t>> &UpdateModelKernel::GetCompletePeriodRecord() {
  std::lock_guard<std::mutex> lock(participation_time_and_num_mtx_);
  return participation_time_and_num_;
}

void UpdateModelKernel::ResetParticipationTimeAndNum() {
  std::lock_guard<std::mutex> lock(participation_time_and_num_mtx_);
  for (auto &it : participation_time_and_num_) {
    it.second = 0;
  }
}

ResultCode UpdateModelKernel::ReachThresholdForUpdateModel(const std::shared_ptr<FBBuilder> &fbb,
                                                           const schema::RequestUpdateModel *update_model_req) {
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    std::string reason = "Current amount for updateModel is enough. Please retry later.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(DEBUG) << reason;
    return ResultCode::kFail;
  }
  return ResultCode::kSuccess;
}

ResultCode UpdateModelKernel::VerifyUpdateModel(const schema::RequestUpdateModel *update_model_req,
                                                const std::shared_ptr<FBBuilder> &fbb, DeviceMeta *device_meta) {
  std::string update_model_fl_id = update_model_req->fl_id()->str();
  auto found = cache::ClientInfos::GetInstance().GetDeviceMeta(update_model_fl_id, device_meta);
  if (!found.IsSuccess()) {
    std::string reason = "devices_meta for " + update_model_fl_id + " is not set. Please retry later.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }
  auto iteration = update_model_req->iteration();
  if (static_cast<uint64_t>(iteration) != cache::InstanceContext::Instance().iteration_num()) {
    auto next_req_time = LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp);
    std::string reason = "UpdateModel iteration number is invalid:" + std::to_string(iteration) +
                         ", current iteration:" + std::to_string(cache::InstanceContext::Instance().iteration_num()) +
                         ", Retry later at time: " + std::to_string(next_req_time) + ", fl id is " + update_model_fl_id;
    BuildUpdateModelRsp(fbb, schema::ResponseCode_OutOfTime, reason, std::to_string(next_req_time));
    MS_LOG(DEBUG) << reason;
    return ResultCode::kFail;
  }

  // verify signature
  if (FLContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(update_model_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed for fl id " + update_model_fl_id;
      BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
      MS_LOG(WARNING) << reason;
      return ResultCode::kFail;
    }

    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason =
        "verify signature timestamp failed or cannot find its key attestation for fl id " + update_model_fl_id;
      BuildUpdateModelRsp(fbb, schema::ResponseCode_OutOfTime, reason, "");
      MS_LOG(WARNING) << reason;
      return ResultCode::kFail;
    }
    MS_LOG(DEBUG) << "verify signature passed!";
  }
  bool verifyFeatureMapIsSuccess = true;
  if (FLContext::instance()->encrypt_type() == kDSEncryptType && update_model_req->sign() != 0) {
    if (update_model_req->index_array() == nullptr) {
      verifyFeatureMapIsSuccess = false;
    } else {
      verifyFeatureMapIsSuccess = VerifySignDSFeatureMap(update_model_req, device_meta);
    }
  } else if (IsCompress(update_model_req)) {
    verifyFeatureMapIsSuccess = VerifyUploadCompressFeatureMap(update_model_req, device_meta);
  } else {
    ModelItemPtr modelItemPtr = ParseModelItemPtr(update_model_req);
    MS_ERROR_IF_NULL_W_RET_VAL(modelItemPtr, ResultCode::kFail);
    verifyFeatureMapIsSuccess = LocalMetaStore::GetInstance().verifyAggregationFeatureMap(modelItemPtr);
  }
  if (!verifyFeatureMapIsSuccess) {
    auto next_req_time = LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp);
    std::string reason = "Verify model feature map failed, retry later at time: " + std::to_string(next_req_time);
    BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, std::to_string(next_req_time));
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }

  if (FLContext::instance()->encrypt_type() == kPWEncryptType) {
    auto find_client = cache::ClientInfos::GetInstance().HasGetSecretsClient(update_model_fl_id);
    if (!find_client) {  // the client not in get_secrets_clients
      std::string reason = "fl_id: " + update_model_fl_id + " is not in get_secrets_clients. Please retry later.";
      BuildUpdateModelRsp(
        fbb, schema::ResponseCode_OutOfTime, reason,
        std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
      MS_LOG(WARNING) << reason;
      return ResultCode::kFail;
    }
  }
  return ResultCode::kSuccess;
}

bool UpdateModelKernel::IsCompress(const schema::RequestUpdateModel *update_model_req) {
  if (FLContext::instance()->compression_config().upload_compress_type != kNoCompressType &&
      update_model_req->upload_compress_type() != schema::CompressType_NO_COMPRESS) {
    return true;
  }
  return false;
}

bool UpdateModelKernel::VerifySignDSFeatureMap(const schema::RequestUpdateModel *update_model_req,
                                               DeviceMeta *device_meta) {
  auto index_array = update_model_req->index_array();
  size_t index_array_size = index_array->size();
  size_t array_size_upper = 100;
  if (index_array_size == 0 || index_array_size > array_size_upper) {
    return false;
  }
  std::map<std::string, std::vector<float>> weight_map;
  std::map<std::string, Address> feature_map =
    ParseSignDSFeatureMap(update_model_req, device_meta->data_size(), &weight_map);
  return LocalMetaStore::GetInstance().verifyAggregationFeatureMap(feature_map);
}

bool UpdateModelKernel::VerifyUploadCompressFeatureMap(const schema::RequestUpdateModel *update_model_req,
                                                       DeviceMeta *device_meta) {
  auto upload_sparse_rate = update_model_req->upload_sparse_rate();
  if (upload_sparse_rate != FLContext::instance()->compression_config().upload_sparse_rate) {
    MS_LOG(WARNING) << "The upload_sparse_rate must be equal to the setting in context.";
    return false;
  }
  auto fbs_name_vec = update_model_req->name_vec();
  if (fbs_name_vec == nullptr) {
    MS_LOG(WARNING) << "The name_vec is null.";
    return false;
  }
  if (fbs_name_vec->size() == 0) {
    MS_LOG(WARNING) << "The size of name_vec must be larger than 0.";
    return false;
  }
  auto fbs_compress_feature_map = update_model_req->compress_feature_map();
  if (fbs_compress_feature_map == nullptr) {
    MS_LOG(WARNING) << "The upload compress feature map is null.";
    return false;
  }
  if (fbs_compress_feature_map->size() == 0) {
    MS_LOG(WARNING) << "The upload compress feature map is empty.";
    return false;
  }
  for (size_t i = 0; i < fbs_compress_feature_map->size(); ++i) {
    int compress_data_size = fbs_compress_feature_map->Get(i)->compress_data()->size();
    if (compress_data_size == 0) {
      continue;
    }

    if (compress_data_size < 0) {
      MS_LOG(WARNING) << "Compress data size must be >= 0.";
      return false;
    }

    float min_val = fbs_compress_feature_map->Get(i)->min_val();
    float max_val = fbs_compress_feature_map->Get(i)->max_val();
    if (min_val > max_val) {
      MS_LOG(WARNING) << "Compress mode min val must be <= max val.";
      return false;
    }

    if (std::isnan(min_val) || std::isinf(min_val)) {
      MS_LOG(WARNING) << "The compress min val is nan or inf.";
      return false;
    }
    if (std::isnan(max_val) || std::isinf(max_val)) {
      MS_LOG(WARNING) << "The compress max val is nan or inf.";
      return false;
    }
  }

  std::map<std::string, std::vector<float>> weight_map;
  std::map<std::string, Address> feature_map =
    ParseUploadCompressFeatureMap(update_model_req, device_meta->data_size(), &weight_map);
  return LocalMetaStore::GetInstance().verifyAggregationFeatureMap(feature_map);
}

ResultCode UpdateModelKernel::ParseAndVerifyFeatureMap(const schema::RequestUpdateModel *update_model_req,
                                                       const DeviceMeta &device_meta,
                                                       const std::shared_ptr<FBBuilder> &fbb,
                                                       std::map<std::string, std::vector<float>> *weight_map_ptr,
                                                       std::map<std::string, Address> *feature_map_ptr) {
  std::string update_model_fl_id = update_model_req->fl_id()->str();
  size_t data_size = device_meta.data_size();

  std::map<std::string, std::vector<float>> &weight_map = *weight_map_ptr;
  std::map<std::string, Address> &feature_map = *feature_map_ptr;
  if (FLContext::instance()->encrypt_type() == kDSEncryptType) {
    feature_map = ParseSignDSFeatureMap(update_model_req, data_size, &weight_map);
  } else if (FLContext::instance()->compression_config().upload_compress_type == kDiffSparseQuant) {
    feature_map = ParseUploadCompressFeatureMap(update_model_req, data_size, &weight_map);
  } else {
    feature_map = ParseFeatureMap(update_model_req);
  }
  if (feature_map.empty()) {
    std::string reason = "Feature map is empty for fl id " + update_model_fl_id;
    BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }
  auto status = Executor::GetInstance().CheckUpdatedModel(feature_map, update_model_fl_id);
  if (!status.IsSuccess()) {
    std::string reason = status.StatusMessage();
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_RequestError, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    return ResultCode::kFail;
  }
  return ResultCode::kSuccess;
}

ResultCode UpdateModelKernel::UpdateModel(const schema::RequestUpdateModel *update_model_req,
                                          const std::shared_ptr<FBBuilder> &fbb, const DeviceMeta &device_meta,
                                          const std::map<std::string, Address> &feature_map) {
  std::string update_model_fl_id = update_model_req->fl_id()->str();
  MS_LOG(DEBUG) << "UpdateModel for fl id " << update_model_fl_id;

  size_t data_size = device_meta.data_size();
  size_t eval_data_size = device_meta.eval_data_size();

  if (!cache::ClientInfos::GetInstance().AddUpdateModelClient(update_model_fl_id).IsSuccess()) {
    std::string reason = "Updating metadata of UpdateModelClientList failed for fl id " + update_model_fl_id;
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }
  if (!DistributedCountService::GetInstance().Count(name_)) {
    std::string reason = "Counting for update model request failed for fl id " + update_model_req->fl_id()->str() +
                         ", Please retry later.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }
  executor_->HandleModelUpdate(feature_map, data_size);
  UpdateClientUploadLoss(update_model_req->upload_loss(), data_size);
  UpdateClientUploadAccuracy(update_model_req->upload_accuracy(), eval_data_size);
  BuildUpdateModelRsp(fbb, schema::ResponseCode_SUCCEED, "success not ready",
                      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
  return ResultCode::kSuccess;
}

ResultCode UpdateModelKernel::CountForAggregation() {
  if (!DistributedCountService::GetInstance().Count(kCountForAggregation)) {
    MS_LOG(ERROR) << "Counting for aggregation failed.";
    return ResultCode::kFail;
  }
  return ResultCode::kSuccess;
}

ModelItemPtr UpdateModelKernel::ParseModelItemPtr(const schema::RequestUpdateModel *update_model_req) {
  ModelItemPtr modelItemPtr = std::make_shared<ModelItem>();
  auto upload_feature_map = update_model_req->feature_map();
  for (uint32_t i = 0; i < upload_feature_map->size(); i++) {
    const auto &item = upload_feature_map->Get(i);

    if (item == nullptr || item->weight_fullname() == nullptr || item->data() == nullptr) {
      MS_LOG(WARNING) << "Verify upload feature map failed";
      return nullptr;
    }
    WeightItem weight;
    std::string weight_full_name = item->weight_fullname()->str();
    const auto &data = item->data();
    size_t weight_size = data->size() * sizeof(float);

    weight.name = weight_full_name;
    weight.size = weight_size;
    uint8_t *upload_weight_data_arr = reinterpret_cast<uint8_t *>(const_cast<float *>(data->data()));
    std::vector<uint8_t> upload_weight_data(upload_weight_data_arr,
                                            upload_weight_data_arr + weight_size / sizeof(uint8_t));
    modelItemPtr->weight_items[weight_full_name] = weight;
    modelItemPtr->weight_data.insert(modelItemPtr->weight_data.end(), upload_weight_data.begin(),
                                     upload_weight_data.end());
  }
  return modelItemPtr;
}

std::map<std::string, Address> UpdateModelKernel::ParseFeatureMap(const schema::RequestUpdateModel *update_model_req) {
  std::map<std::string, Address> feature_map;
  auto fbs_feature_map = update_model_req->feature_map();
  for (uint32_t i = 0; i < fbs_feature_map->size(); i++) {
    auto feature = fbs_feature_map->Get(i);
    if (feature == nullptr || feature->weight_fullname() == nullptr || feature->data() == nullptr) {
      MS_LOG_WARNING << "Feature parsed from flatbuffer is invalid";
      return {};
    }
    std::string weight_full_name = feature->weight_fullname()->str();
    auto weight_data = feature->data()->data();
    size_t weight_size = feature->data()->size() * sizeof(float);
    Address upload_data;
    upload_data.addr = weight_data;
    upload_data.size = weight_size;
    feature_map[weight_full_name] = upload_data;
  }
  return feature_map;
}

std::map<std::string, Address> UpdateModelKernel::ParseSignDSFeatureMap(
  const schema::RequestUpdateModel *update_model_req, size_t data_size,
  std::map<std::string, std::vector<float>> *weight_map) {
  auto fbs_feature_map = update_model_req->feature_map();
  std::map<std::string, Address> feature_map;
  auto sign = update_model_req->sign();
  if (sign == 0) {
    feature_map = ParseFeatureMap(update_model_req);
    return feature_map;
  }

  auto latest_model = ModelStore::GetInstance().GetLatestModel().second;
  if (latest_model == nullptr || latest_model->weight_data.empty()) {
    MS_LOG_ERROR << "Failed to get latest model";
    return {};
  }
  auto weight_data = latest_model->weight_data.data();
  auto index_array = update_model_req->index_array();
  size_t index_store = 0;
  size_t index_array_j = 0;
  float signds_grad = sign * FLContext::instance()->encrypt_config().sign_global_lr;
  for (size_t i = 0; i < fbs_feature_map->size(); i++) {
    std::string weight_full_name = fbs_feature_map->Get(i)->weight_fullname()->str();
    auto weight_info = latest_model->weight_items[weight_full_name];
    auto feature_data = weight_data + weight_info.offset;
    size_t iter_feature_num = weight_info.size / sizeof(float);
    auto &weight_item = (*weight_map)[weight_full_name];
    weight_item.resize(iter_feature_num);
    float *iter_feature_map_data = reinterpret_cast<float *>(feature_data);
    for (size_t j = 0; j < iter_feature_num; j++) {
      float reconstruct_weight = iter_feature_map_data[j];
      if (index_array_j < index_array->size() && index_store == static_cast<size_t>(index_array->Get(index_array_j))) {
        reconstruct_weight += signds_grad;
        index_array_j++;
      }
      reconstruct_weight *= data_size;
      index_store++;
      weight_item[j] = reconstruct_weight;
    }
    size_t weight_size = iter_feature_num * sizeof(float);
    Address upload_data;
    upload_data.addr = weight_item.data();
    upload_data.size = weight_size;
    feature_map[weight_full_name] = upload_data;
  }
  return feature_map;
}

std::map<std::string, Address> UpdateModelKernel::ParseUploadCompressFeatureMap(
  const schema::RequestUpdateModel *update_model_req, size_t data_size,
  std::map<std::string, std::vector<float>> *weight_map) {
  std::map<std::string, Address> feature_map;
  schema::CompressType upload_compress_type = update_model_req->upload_compress_type();
  upload_compress_type =
    mindspore::fl::compression::DecodeExecutor::GetInstance().GetCompressType(upload_compress_type);
  MS_LOG(DEBUG) << "This schema upload compress type is: " << upload_compress_type;
  if (upload_compress_type != schema::CompressType_NO_COMPRESS) {
    MS_LOG(DEBUG) << "This upload compress type is DIFF_SPARSE_QUANT.";
    feature_map = DecodeFeatureMap(weight_map, update_model_req, upload_compress_type, data_size);
    return feature_map;
  }
  MS_LOG(DEBUG) << "This upload compress type is NO_COMPRESS.";
  // Some clients upload origin weights.
  auto fbs_feature_map = update_model_req->feature_map();
  for (uint32_t i = 0; i < fbs_feature_map->size(); i++) {
    std::string weight_full_name = fbs_feature_map->Get(i)->weight_fullname()->str();
    float *weight_data = const_cast<float *>(fbs_feature_map->Get(i)->data()->data());
    size_t weight_size = fbs_feature_map->Get(i)->data()->size() * sizeof(float);
    Address upload_data;
    upload_data.addr = weight_data;
    upload_data.size = weight_size;
    feature_map[weight_full_name] = upload_data;
  }
  return feature_map;
}

std::map<std::string, Address> UpdateModelKernel::DecodeFeatureMap(
  std::map<std::string, std::vector<float>> *weight_map, const schema::RequestUpdateModel *update_model_req,
  schema::CompressType upload_compress_type, size_t data_size) {
  std::map<std::string, Address> feature_map;

  // Get and set decode hyper parameters.
  auto seed = update_model_req->iteration();
  MS_LOG(DEBUG) << "The seed for compression is: " << seed;
  auto upload_sparse_rate = update_model_req->upload_sparse_rate();
  MS_LOG(DEBUG) << "The upload_sparse_rate for compression is: " << upload_sparse_rate;
  // Get name vector.
  auto fbs_name_vec = update_model_req->name_vec();
  std::vector<std::string> name_vec;
  for (size_t i = 0; i < fbs_name_vec->size(); ++i) {
    name_vec.emplace_back(fbs_name_vec->Get(i)->str());
  }
  // Parameter process for decode.
  auto fbs_compress_feature_map = update_model_req->compress_feature_map();
  std::vector<mindspore::fl::compression::CompressFeatureMap> compress_feature_maps;
  for (size_t i = 0; i < fbs_compress_feature_map->size(); ++i) {
    mindspore::fl::compression::CompressFeatureMap compress_feature_map;
    auto feature = fbs_compress_feature_map->Get(i);
    int8_t *compress_weight_data = const_cast<int8_t *>(feature->compress_data()->data());
    size_t compress_weight_size = feature->compress_data()->size();
    MS_LOG(DEBUG) << "The compress weight size: " << compress_weight_size;
    for (size_t j = 0; j < compress_weight_size; ++j) {
      compress_feature_map.compress_data.emplace_back(compress_weight_data[j]);
    }
    compress_feature_map.min_val = feature->min_val();
    compress_feature_map.max_val = feature->max_val();
    MS_LOG(DEBUG) << "Min value: " << compress_feature_map.min_val;
    MS_LOG(DEBUG) << "Max value: " << compress_feature_map.max_val;
    compress_feature_maps.emplace_back(compress_feature_map);
  }

  // Decode.
  bool status = mindspore::fl::compression::DecodeExecutor::GetInstance().Decode(
    weight_map, compress_feature_maps, upload_compress_type, upload_sparse_rate, seed, name_vec, data_size);
  if (status) {
    for (size_t i = 0; i < name_vec.size(); ++i) {
      std::string weight_full_name = name_vec[i];
      size_t weight_size = (*weight_map)[weight_full_name].size() * sizeof(float);
      Address upload_data;
      upload_data.addr = (*weight_map)[weight_full_name].data();
      upload_data.size = weight_size;
      feature_map[weight_full_name] = upload_data;
    }
    return feature_map;
  }
  MS_LOG(WARNING) << "Decode failed!";
  return feature_map;
}

sigVerifyResult UpdateModelKernel::VerifySignature(const schema::RequestUpdateModel *update_model_req) {
  return VerifySignatureBase(update_model_req);
}

void UpdateModelKernel::BuildUpdateModelRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                            const std::string &reason, const std::string &next_req_time) {
  if (fbb == nullptr) {
    MS_LOG(WARNING) << "Input fbb is nullptr.";
    return;
  }
  auto fbs_reason = fbb->CreateString(reason);
  auto fbs_next_req_time = fbb->CreateString(next_req_time);

  schema::ResponseUpdateModelBuilder rsp_update_model_builder(*(fbb.get()));
  rsp_update_model_builder.add_retcode(static_cast<int>(retcode));
  rsp_update_model_builder.add_reason(fbs_reason);
  rsp_update_model_builder.add_next_req_time(fbs_next_req_time);
  auto rsp_update_model = rsp_update_model_builder.Finish();
  fbb->Finish(rsp_update_model);
  return;
}

void UpdateModelKernel::RecordCompletePeriod(const DeviceMeta &device_meta) {
  std::lock_guard<std::mutex> lock(participation_time_and_num_mtx_);
  uint64_t start_fl_job_time = device_meta.now_time();
  uint64_t update_model_complete_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  if (start_fl_job_time >= update_model_complete_time) {
    MS_LOG(WARNING) << "start_fl_job_time " << start_fl_job_time << " is larger than update_model_complete_time "
                    << update_model_complete_time;
    return;
  }
  uint64_t cost_time = update_model_complete_time - start_fl_job_time;
  MS_LOG(DEBUG) << "start_fl_job time  is " << start_fl_job_time << " update_model time is "
                << update_model_complete_time;
  for (auto &it : participation_time_and_num_) {
    if (cost_time < it.first) {
      it.second++;
    }
  }
}

void UpdateModelKernel::CheckAndTransPara(const std::string &participation_time_level) {
  std::lock_guard<std::mutex> lock(participation_time_and_num_mtx_);
  // The default time level is 5min and 15min, trans time to millisecond
  participation_time_and_num_.emplace_back(std::make_pair(kDefaultLevel1 * kMinuteToSecond * kSecondToMills, 0));
  participation_time_and_num_.emplace_back(std::make_pair(kDefaultLevel2 * kMinuteToSecond * kSecondToMills, 0));
  participation_time_and_num_.emplace_back(std::make_pair(UINT64_MAX, 0));
  std::vector<std::string> time_levels;
  std::istringstream iss(participation_time_level);
  std::string output;
  while (std::getline(iss, output, ',')) {
    if (!output.empty()) {
      time_levels.emplace_back(std::move(output));
    }
  }
  if (time_levels.size() != kLevelNum) {
    MS_LOG(WARNING) << "Parameter participation_time_level is not correct";
    return;
  }
  uint64_t level1 = std::strtoull(time_levels[0].c_str(), nullptr, kBase);
  if (level1 > kMaxLevelNum || level1 <= kMinLevelNum) {
    MS_LOG(WARNING) << "Level1 partmeter " << level1 << " is not legal";
    return;
  }

  uint64_t level2 = std::strtoull(time_levels[1].c_str(), nullptr, kBase);
  if (level2 > kMaxLevelNum || level2 <= kMinLevelNum) {
    MS_LOG(WARNING) << "Level2 partmeter " << level2 << "is not legal";
    return;
  }
  if (level1 >= level2) {
    MS_LOG(WARNING) << "Level1 parameter " << level1 << " is larger than level2 " << level2;
    return;
  }
  // Save the the parament of user
  participation_time_and_num_.clear();
  participation_time_and_num_.emplace_back(std::make_pair(level1 * kMinuteToSecond * kSecondToMills, 0));
  participation_time_and_num_.emplace_back(std::make_pair(level2 * kMinuteToSecond * kSecondToMills, 0));
  participation_time_and_num_.emplace_back(std::make_pair(UINT64_MAX, 0));
}

REG_ROUND_KERNEL(updateModel, UpdateModelKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
