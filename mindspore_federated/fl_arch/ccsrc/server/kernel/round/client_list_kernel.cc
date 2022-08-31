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

#include "server/kernel/round/client_list_kernel.h"
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include "schema/cipher_generated.h"
#include "distributed_cache/client_infos.h"
#include "distributed_cache/counter.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void ClientListKernel::InitKernel(size_t) { cipher_init_ = &armour::CipherInit::GetInstance(); }

sigVerifyResult ClientListKernel::VerifySignature(const schema::GetClientList *get_clients_req) {
  return VerifySignatureBase(get_clients_req);
}

bool ClientListKernel::DealClient(const size_t iter_num, const schema::GetClientList *get_clients_req,
                                  const std::shared_ptr<FBBuilder> &fbb) {
  std::vector<std::string> empty_client_list;
  std::string fl_id = get_clients_req->fl_id()->str();

  if (!LocalMetaStore::GetInstance().has_value(kCtxUpdateModelThld)) {
    MS_LOG(ERROR) << "update_model_client_threshold is not set.";
    BuildClientListRsp(fbb, schema::ResponseCode_SystemError, "update_model_client_threshold is not set.",
                       empty_client_list, std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    return false;
  }
  uint64_t update_model_client_needed = LocalMetaStore::GetInstance().value<uint64_t>(kCtxUpdateModelThld);
  bool updateModelOK = DistributedCountService::GetInstance().CountReachThreshold("updateModel");
  if (!updateModelOK) {
    MS_LOG(INFO) << "The server is not ready. update_model_client_needed: " << update_model_client_needed;
    BuildClientListRsp(fbb, schema::ResponseCode_SucNotReady, "The server is not ready.", empty_client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    return false;
  }

  auto found = fl::cache::ClientInfos::GetInstance().HasUpdateModelClient(fl_id);
  if (!found) {
    std::string reason = "fl_id: " + fl_id + " is not in the update_model_clients";
    MS_LOG(INFO) << reason;
    BuildClientListRsp(fbb, schema::ResponseCode_RequestError, reason, empty_client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    return false;
  }

  auto status = cache::ClientInfos::GetInstance().AddGetUpdateModelClient(fl_id);
  if (!status.IsSuccess()) {
    std::string reason = "update get update model clients failed";
    MS_LOG(ERROR) << reason;
    BuildClientListRsp(fbb, schema::ResponseCode_SucNotReady, reason, empty_client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    return false;
  }
  if (!DistributedCountService::GetInstance().Count(name_)) {
    std::string reason = "Counting for get user list request failed. Please retry later.";
    BuildClientListRsp(fbb, schema::ResponseCode_OutOfTime, reason, empty_client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    MS_LOG(ERROR) << reason;
    return false;
  }
  MS_LOG(INFO) << "update_model_client_needed: " << update_model_client_needed;
  std::vector<std::string> client_list;
  auto ret = cache::ClientInfos::GetInstance().GetAllUpdateModelClients(&client_list);
  if (!ret.IsSuccess()) {
    MS_LOG(INFO) << "The server is not ready. get update model client list failed";
    BuildClientListRsp(fbb, schema::ResponseCode_SucNotReady, "The server is not ready.", empty_client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    return false;
  }
  BuildClientListRsp(fbb, schema::ResponseCode_SUCCEED, "send clients_list succeed!", client_list,
                     std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
  return true;
}

bool ClientListKernel::Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) {
  size_t iter_num = cache::InstanceContext::Instance().iteration_num();
  MS_LOG(INFO) << "Launching ClientListKernel, Iteration number is " << iter_num;

  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    return false;
  }
  std::vector<std::string> client_list;
  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::GetClientList>()) {
    std::string reason = "The schema of GetClientList is invalid.";
    BuildClientListRsp(fbb, schema::ResponseCode_RequestError, reason, client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  const schema::GetClientList *get_clients_req = flatbuffers::GetRoot<schema::GetClientList>(req_data);
  if (get_clients_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for GetClientList.";
    BuildClientListRsp(fbb, schema::ResponseCode_RequestError, reason, client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  // verify signature
  if (FLContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(get_clients_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed.";
      BuildClientListRsp(fbb, schema::ResponseCode_RequestError, reason, client_list,
                         std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }
    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed or cannot find its key attestation.";
      BuildClientListRsp(fbb, schema::ResponseCode_OutOfTime, reason, client_list,
                         std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }
    MS_LOG(DEBUG) << "verify signature passed!";
  }

  size_t iter_client = IntToSize(get_clients_req->iteration());
  if (iter_num != iter_client) {
    MS_LOG(ERROR) << "client list iteration number is invalid: server now iteration is " << iter_num
                  << ". client request iteration is " << iter_client;
    BuildClientListRsp(fbb, schema::ResponseCode_OutOfTime, "iter num is error.", client_list,
                       std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)),
                       iter_num);
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(WARNING) << "Current amount for GetClientList is enough.";
  }

  if (!DealClient(iter_num, get_clients_req, fbb)) {
    MS_LOG(WARNING) << "Get Client List not ready.";
  }
  SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
  return true;
}  // namespace fl

void ClientListKernel::BuildClientListRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                          const std::string &reason, std::vector<std::string> clients,
                                          const std::string &next_req_time, const size_t iteration) {
  auto rsp_reason = fbb->CreateString(reason);
  auto rsp_next_req_time = fbb->CreateString(next_req_time);
  std::vector<flatbuffers::Offset<flatbuffers::String>> clients_vector;
  for (auto client : clients) {
    auto client_fb = fbb->CreateString(client);
    clients_vector.push_back(client_fb);
    MS_LOG(WARNING) << "update client list: ";
    MS_LOG(WARNING) << client;
  }
  auto clients_fb = fbb->CreateVector(clients_vector);
  schema::ReturnClientListBuilder rsp_builder(*(fbb.get()));
  rsp_builder.add_retcode(SizeToInt(retcode));
  rsp_builder.add_reason(rsp_reason);
  rsp_builder.add_clients(clients_fb);
  rsp_builder.add_iteration(SizeToInt(iteration));
  rsp_builder.add_next_req_time(rsp_next_req_time);
  auto rsp_exchange_keys = rsp_builder.Finish();
  fbb->Finish(rsp_exchange_keys);
  return;
}

REG_ROUND_KERNEL(getClientList, ClientListKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
