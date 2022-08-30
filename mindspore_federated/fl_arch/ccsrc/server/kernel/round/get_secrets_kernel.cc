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

#include "server/kernel/round/get_secrets_kernel.h"
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <utility>
#include "armour/cipher/cipher_shares.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void GetSecretsKernel::InitKernel(size_t) { cipher_share_ = &armour::CipherShares::GetInstance(); }

bool GetSecretsKernel::CountForGetSecrets(const std::shared_ptr<FBBuilder> &fbb,
                                          const schema::GetShareSecrets *get_secrets_req, const size_t iter_num) {
  MS_ERROR_IF_NULL_W_RET_VAL(get_secrets_req, false);
  auto fbs_fl_id = get_secrets_req->fl_id();
  MS_EXCEPTION_IF_NULL(fbs_fl_id);
  if (!DistributedCountService::GetInstance().Count(name_)) {
    std::string reason = "Counting for get secrets kernel request failed. Please retry later.";
    cipher_share_->BuildGetSecretsRsp(fbb, schema::ResponseCode_OutOfTime, iter_num,
                                      std::to_string(CURRENT_TIME_MILLI.count()), nullptr);
    MS_LOG(ERROR) << reason;
    return false;
  }
  return true;
}

sigVerifyResult GetSecretsKernel::VerifySignature(const schema::GetShareSecrets *get_secrets_req) {
  return VerifySignatureBase(get_secrets_req);
}

bool GetSecretsKernel::Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) {
  size_t iter_num = cache::InstanceContext::Instance().iteration_num();
  std::string next_timestamp = std::to_string(CURRENT_TIME_MILLI.count());
  MS_LOG(INFO) << "Launching get secrets kernel, ITERATION NUMBER IS : " << iter_num;
  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    return false;
  }
  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::GetShareSecrets>()) {
    std::string reason = "The schema of GetShareSecrets is invalid.";
    cipher_share_->BuildGetSecretsRsp(fbb, schema::ResponseCode_RequestError, iter_num, next_timestamp, nullptr);
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  const schema::GetShareSecrets *get_secrets_req = flatbuffers::GetRoot<schema::GetShareSecrets>(req_data);
  if (get_secrets_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for GetExchangeKeys.";
    cipher_share_->BuildGetSecretsRsp(fbb, schema::ResponseCode_RequestError, iter_num, next_timestamp, nullptr);
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  // verify signature
  if (FLContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(get_secrets_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed.";
      cipher_share_->BuildGetSecretsRsp(fbb, schema::ResponseCode_RequestError, iter_num, next_timestamp, nullptr);
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed or cannot find its key attestation.";
      cipher_share_->BuildGetSecretsRsp(fbb, schema::ResponseCode_OutOfTime, iter_num, next_timestamp, nullptr);
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    if (verify_result == sigVerifyResult::PASSED) {
      MS_LOG(INFO) << "verify signature passed!";
    }
  }
  size_t iter_client = IntToSize(get_secrets_req->iteration());
  if (iter_num != iter_client) {
    MS_LOG(ERROR) << "GetSecretsKernel iteration invalid. server now iteration is " << iter_num
                  << ". client request iteration is " << iter_client;
    cipher_share_->BuildGetSecretsRsp(fbb, schema::ResponseCode_OutOfTime, iter_num, next_timestamp, nullptr);
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(ERROR) << "Current amount for GetSecretsKernel is enough.";
  }

  bool response = cipher_share_->GetSecrets(get_secrets_req, fbb, next_timestamp);
  if (!response) {
    MS_LOG(WARNING) << "get secret shares not ready.";
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  if (!CountForGetSecrets(fbb, get_secrets_req, iter_num)) {
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
  return true;
}

REG_ROUND_KERNEL(getSecrets, GetSecretsKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
