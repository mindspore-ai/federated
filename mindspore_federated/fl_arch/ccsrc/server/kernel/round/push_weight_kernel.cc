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

#include "server/kernel/round/push_weight_kernel.h"
#include "server/server.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void PushWeightKernel::InitKernel(size_t) {}

bool PushWeightKernel::Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) {
  MS_LOG(INFO) << "Launching PushWeightKernel kernel.";
  FBBuilder fbb;
  auto current_iter = cache::InstanceContext::Instance().iteration_num();
  auto fl_status = OnReceiveModelWeight(req_data, len);
  if (!fl_status.IsSuccess()) {
    BuildPushWeightRsp(&fbb, schema::ResponseCode_SucNotReady, fl_status.StatusMessage(), current_iter);
    SendResponseMsg(message, fbb.GetBufferPointer(), fbb.GetSize());
    return true;
  }
  BuildPushWeightRsp(&fbb, schema::ResponseCode_SUCCEED, "PushWeight succeed.", current_iter);
  SendResponseMsg(message, fbb.GetBufferPointer(), fbb.GetSize());
  MS_LOG(INFO) << "Launching PushWeightKernel successful.";
  return true;
}

FlStatus PushWeightKernel::OnReceiveModelWeight(const uint8_t *req_data, size_t len) {
  std::string reason;
  auto current_iter = cache::InstanceContext::Instance().iteration_num();
  if (Executor::GetInstance().IsIterationModelFinished(current_iter)) {
    reason = "Model of iteration of " + std::to_string(current_iter) + " has finished";
    MS_LOG(INFO) << reason;
    return {kRequestError, reason};
  }
  if (req_data == nullptr) {
    reason = "req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    return {kRequestError, reason};
  }
  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::RequestPushWeight>()) {
    reason = "The schema of RequestPushWeight is invalid.";
    MS_LOG(ERROR) << reason;
    return {kRequestError, reason};
  }
  const schema::RequestPushWeight *push_weight_req = flatbuffers::GetRoot<schema::RequestPushWeight>(req_data);
  if (push_weight_req == nullptr) {
    reason = "Building flatbuffers schema failed for RequestPushWeight";
    MS_LOG(ERROR) << reason;
    return {kRequestError, reason};
  }
  uint64_t iteration = static_cast<uint64_t>(push_weight_req->iteration());
  if (iteration != current_iter) {
    reason = "PushWeight iteration number is invalid:" + std::to_string(iteration) +
             ", current iteration:" + std::to_string(current_iter);
    MS_LOG(ERROR) << reason;
    return {kRequestError, reason};
  }
  std::map<std::string, Address> upload_feature_map = ParseFeatureMap(push_weight_req);
  if (upload_feature_map.empty()) {
    reason = "PushWeight feature map is empty.";
    MS_LOG(ERROR) << reason;
    return {kRequestError, reason};
  }
  if (!Executor::GetInstance().HandlePushWeight(upload_feature_map)) {
    reason = "Pushing weight failed.";
    MS_LOG(ERROR) << reason;
    return {kRequestError, reason};
  }
  return kFlSuccess;
}

std::map<std::string, Address> PushWeightKernel::ParseFeatureMap(const schema::RequestPushWeight *push_weight_req) {
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

void PushWeightKernel::BuildPushWeightRsp(FBBuilder *fbb, const schema::ResponseCode retcode, const std::string &reason,
                                          size_t iteration) {
  if (fbb == nullptr) {
    MS_LOG(ERROR) << "Input fbb is nullptr.";
    return;
  }
  auto fbs_reason = fbb->CreateString(reason);
  schema::ResponsePushWeightBuilder rsp_push_weight_builder(*fbb);
  rsp_push_weight_builder.add_retcode(SizeToInt(retcode));
  rsp_push_weight_builder.add_reason(fbs_reason);
  rsp_push_weight_builder.add_iteration(SizeToInt(iteration));
  auto rsp_push_weight = rsp_push_weight_builder.Finish();
  fbb->Finish(rsp_push_weight);
}

REG_ROUND_KERNEL(pushWeight, PushWeightKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
