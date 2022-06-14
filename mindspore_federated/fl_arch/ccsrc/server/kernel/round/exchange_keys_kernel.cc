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

#include "server/kernel/round/exchange_keys_kernel.h"
#include <vector>
#include <utility>
#include <memory>
#include <map>

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void ExchangeKeysKernel::InitKernel(size_t) { cipher_key_ = &armour::CipherKeys::GetInstance(); }

bool ExchangeKeysKernel::ReachThresholdForExchangeKeys(const std::shared_ptr<FBBuilder> &fbb, const size_t iter_num) {
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    std::string reason = "Current amount for exchangeKey is enough. Please retry later.";
    cipher_key_->BuildExchangeKeysRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)), iter_num);
    MS_LOG(WARNING) << reason;
    return true;
  }
  return false;
}

bool ExchangeKeysKernel::CountForExchangeKeys(const std::shared_ptr<FBBuilder> &fbb,
                                              const schema::RequestExchangeKeys *exchange_keys_req,
                                              const size_t iter_num) {
  MS_ERROR_IF_NULL_W_RET_VAL(exchange_keys_req, false);
  if (!DistributedCountService::GetInstance().Count(name_)) {
    std::string reason = "Counting for exchange kernel request failed. Please retry later.";
    cipher_key_->BuildExchangeKeysRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)), iter_num);
    MS_LOG(ERROR) << reason;
    return false;
  }
  return true;
}

sigVerifyResult ExchangeKeysKernel::VerifySignature(const schema::RequestExchangeKeys *exchange_keys_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(exchange_keys_req, sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(exchange_keys_req->fl_id(), sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(exchange_keys_req->timestamp(), sigVerifyResult::FAILED);

  std::string fl_id = exchange_keys_req->fl_id()->str();
  std::string timestamp = exchange_keys_req->timestamp()->str();
  int iteration = exchange_keys_req->iteration();
  std::string iter_str = std::to_string(iteration);
  auto fbs_cpk = exchange_keys_req->c_pk();
  auto fbs_spk = exchange_keys_req->s_pk();
  if (fbs_cpk == nullptr || fbs_spk == nullptr) {
    MS_LOG(ERROR) << "public key from exchange_keys_req is null";
    return sigVerifyResult::FAILED;
  }
  std::vector<uint8_t> src_data;
  (void)src_data.insert(src_data.end(), fbs_cpk->begin(), fbs_cpk->end());
  (void)src_data.insert(src_data.end(), fbs_spk->begin(), fbs_spk->end());
  (void)src_data.insert(src_data.end(), timestamp.begin(), timestamp.end());
  (void)src_data.insert(src_data.end(), iter_str.begin(), iter_str.end());

  auto fbs_signature = exchange_keys_req->signature();
  return VerifySignatureBase(fl_id, src_data, fbs_signature, timestamp);
}

bool ExchangeKeysKernel::Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) {
  size_t iter_num = cache::InstanceContext::Instance().iteration_num();
  MS_LOG(INFO) << "Launching ExchangeKey kernel, ITERATION NUMBER IS : " << iter_num;
  bool response = false;

  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    return false;
  }

  if (ReachThresholdForExchangeKeys(fbb, iter_num)) {
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::RequestExchangeKeys>()) {
    std::string reason = "The schema of RequestExchangeKeys is invalid.";
    cipher_key_->BuildExchangeKeysRsp(fbb, schema::ResponseCode_RequestError, reason,
                                      std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  const schema::RequestExchangeKeys *exchange_keys_req = flatbuffers::GetRoot<schema::RequestExchangeKeys>(req_data);
  if (exchange_keys_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for ExchangeKeys.";
    cipher_key_->BuildExchangeKeysRsp(fbb, schema::ResponseCode_RequestError, reason,
                                      std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  // verify signature
  if (FLContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(exchange_keys_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed.";
      cipher_key_->BuildExchangeKeysRsp(fbb, schema::ResponseCode_RequestError, reason,
                                        std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed.";
      cipher_key_->BuildExchangeKeysRsp(fbb, schema::ResponseCode_OutOfTime, reason,
                                        std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    MS_LOG(INFO) << "verify signature passed!";
  }

  size_t iter_client = IntToSize(exchange_keys_req->iteration());
  if (iter_num != iter_client) {
    MS_LOG(ERROR) << "ExchangeKeys iteration number is invalid: server now iteration is " << iter_num
                  << ". client request iteration is " << iter_client;
    cipher_key_->BuildExchangeKeysRsp(fbb, schema::ResponseCode_OutOfTime, "iter num is error.",
                                      std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  response = cipher_key_->ExchangeKeys(iter_num, std::to_string(CURRENT_TIME_MILLI.count()), exchange_keys_req, fbb);
  if (!response) {
    MS_LOG(ERROR) << "update exchange keys is failed.";
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  if (!CountForExchangeKeys(fbb, exchange_keys_req, iter_num)) {
    MS_LOG(ERROR) << "count for exchange keys failed.";
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
  return true;
}

REG_ROUND_KERNEL(exchangeKeys, ExchangeKeysKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
