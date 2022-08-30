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

#include "server/kernel/round/push_list_sign_kernel.h"
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include "schema/cipher_generated.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void PushListSignKernel::InitKernel(size_t) { cipher_init_ = &armour::CipherInit::GetInstance(); }

bool PushListSignKernel::Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) {
  size_t iter_num = cache::InstanceContext::Instance().iteration_num();
  MS_LOG(INFO) << "Launching PushListSignKernel, Iteration number is " << iter_num;
  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    return false;
  }
  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::SendClientListSign>()) {
    std::string reason = "The schema of PushClientListSign is invalid.";
    BuildPushListSignKernelRsp(fbb, schema::ResponseCode_RequestError, reason,
                               std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  const schema::SendClientListSign *client_list_sign_req = flatbuffers::GetRoot<schema::SendClientListSign>(req_data);
  if (client_list_sign_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for PushClientListSign.";
    BuildPushListSignKernelRsp(fbb, schema::ResponseCode_RequestError, reason,
                               std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  // verify signature
  if (FLContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(client_list_sign_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed.";
      BuildPushListSignKernelRsp(fbb, schema::ResponseCode_RequestError, reason,
                                 std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }
    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed or cannot find its key attestation.";
      BuildPushListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason,
                                 std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }
    MS_LOG(INFO) << "verify signature passed!";
  }
  return LaunchForPushListSign(client_list_sign_req, iter_num, fbb, message);
}

bool PushListSignKernel::LaunchForPushListSign(const schema::SendClientListSign *client_list_sign_req,
                                               const size_t &iter_num, const std::shared_ptr<FBBuilder> &fbb,
                                               const std::shared_ptr<MessageHandler> &message) {
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req, false);
  size_t iter_client = IntToSize(client_list_sign_req->iteration());
  if (iter_num != iter_client) {
    std::string reason = "push list sign iteration number is invalid";
    MS_LOG(WARNING) << reason;
    MS_LOG(WARNING) << "server now iteration is " << iter_num << ". client request iteration is " << iter_client;
    BuildPushListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, std::to_string(CURRENT_TIME_MILLI.count()),
                               iter_num);
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req->fl_id(), false);
  std::string fl_id = client_list_sign_req->fl_id()->str();
  auto found_in_update_model_list = fl::cache::ClientInfos::GetInstance().HasUpdateModelClient(fl_id);
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(ERROR) << "Current amount for PushListSignKernel is enough.";
    if (found_in_update_model_list) {
      // client in get update model client list.
      BuildPushListSignKernelRsp(fbb, schema::ResponseCode_SUCCEED, "Current amount for PushListSignKernel is enough.",
                                 std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    } else {
      BuildPushListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime,
                                 "Current amount for PushListSignKernel is enough.",
                                 std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    }
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  if (!PushListSign(iter_num, std::to_string(CURRENT_TIME_MILLI.count()), client_list_sign_req, fbb,
                    found_in_update_model_list)) {
    MS_LOG(ERROR) << "push client list sign failed.";
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  std::string count_reason = "";
  if (!DistributedCountService::GetInstance().Count(name_)) {
    std::string reason = "Counting for push list sign request failed. Please retry later. " + count_reason;
    BuildPushListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, std::to_string(CURRENT_TIME_MILLI.count()),
                               iter_num);
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
  return true;
}

sigVerifyResult PushListSignKernel::VerifySignature(const schema::SendClientListSign *client_list_sign_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req, sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req->fl_id(), sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req->timestamp(), sigVerifyResult::FAILED);

  std::string fl_id = client_list_sign_req->fl_id()->str();
  std::string timestamp = client_list_sign_req->timestamp()->str();
  int iteration = client_list_sign_req->iteration();
  std::string iter_str = std::to_string(iteration);
  auto fbs_signature = client_list_sign_req->req_signature();
  return VerifySignatureBase(fl_id, {timestamp, iter_str}, fbs_signature, timestamp);
}

bool PushListSignKernel::PushListSign(const size_t cur_iterator, const std::string &next_req_time,
                                      const schema::SendClientListSign *client_list_sign_req,
                                      const std::shared_ptr<fl::FBBuilder> &fbb, bool found_in_update_model_clients) {
  MS_LOG(INFO) << "CipherMgr::PushClientListSign START";
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req, false);
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req->fl_id(), false);

  std::string fl_id = client_list_sign_req->fl_id()->str();
  auto found = cache::ClientInfos::GetInstance().HasGetUpdateModelClient(fl_id);
  if (!found) {
    // client not in get update model client list.
    std::string reason = "client send signature is not in get update model client list. && client is illegal";
    MS_LOG(WARNING) << reason;
    if (found_in_update_model_clients) {
      // client in update model client list, client can move to next round
      BuildPushListSignKernelRsp(fbb, schema::ResponseCode_SUCCEED, reason, next_req_time, cur_iterator);
    } else {
      BuildPushListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, next_req_time, cur_iterator);
    }
    return false;
  }

  std::string client_sign_str;
  auto status = cache::ClientInfos::GetInstance().GetClientListSign(fl_id, &client_sign_str);
  if (status.IsSuccess()) {
    // the client has sended signature, return false.
    std::string reason = "The server has received the request, please do not request again.";
    MS_LOG(ERROR) << reason;
    BuildPushListSignKernelRsp(fbb, schema::ResponseCode_SUCCEED, reason, next_req_time, cur_iterator);
    return false;
  }
  auto fbs_signature = client_list_sign_req->signature();
  std::string signature;
  if (fbs_signature != nullptr) {
    signature.assign(fbs_signature->begin(), fbs_signature->end());
  }
  status = cache::ClientInfos::GetInstance().AddClientListSign(fl_id, signature);
  if (!status.IsSuccess()) {
    std::string reason = "store client list signature failed";
    MS_LOG(ERROR) << reason;
    BuildPushListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, next_req_time, cur_iterator);
    return false;
  }
  std::string reason = "send update model client list signature success. ";
  BuildPushListSignKernelRsp(fbb, schema::ResponseCode_SUCCEED, reason, next_req_time, cur_iterator);
  MS_LOG(INFO) << "CipherMgr::PushClientListSign Success";
  return true;
}

bool PushListSignKernel::Reset() {
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << cache::InstanceContext::Instance().iteration_num();
  MS_LOG(INFO) << "Push list sign kernel reset!";
  return true;
}

void PushListSignKernel::BuildPushListSignKernelRsp(const std::shared_ptr<FBBuilder> &fbb,
                                                    const schema::ResponseCode retcode, const std::string &reason,
                                                    const std::string &next_req_time, const size_t iteration) {
  auto rsp_reason = fbb->CreateString(reason);
  auto rsp_next_req_time = fbb->CreateString(next_req_time);
  schema::ResponseClientListSignBuilder rsp_builder(*(fbb.get()));
  rsp_builder.add_retcode(static_cast<int>(retcode));
  rsp_builder.add_reason(rsp_reason);
  rsp_builder.add_next_req_time(rsp_next_req_time);
  rsp_builder.add_iteration(SizeToInt(iteration));
  auto rsp_push_list_sign = rsp_builder.Finish();
  fbb->Finish(rsp_push_list_sign);
  return;
}

REG_ROUND_KERNEL(pushListSign, PushListSignKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
