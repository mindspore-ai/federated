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

#include "server/kernel/round/get_list_sign_kernel.h"
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
void GetListSignKernel::InitKernel(size_t) { cipher_init_ = &armour::CipherInit::GetInstance(); }

sigVerifyResult GetListSignKernel::VerifySignature(const schema::RequestAllClientListSign *client_list_sign_req) {
  return VerifySignatureBase(client_list_sign_req);
}

bool GetListSignKernel::Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) {
  size_t iter_num = cache::InstanceContext::Instance().iteration_num();
  MS_LOG(INFO) << "Launching GetListSign kernel,  Iteration number is " << iter_num;
  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    return false;
  }
  std::map<std::string, std::vector<unsigned char>> list_signs;
  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::RequestAllClientListSign>()) {
    std::string reason = "The schema of RequestAllClientListSign is invalid.";
    BuildGetListSignKernelRsp(fbb, schema::ResponseCode_RequestError, reason,
                              std::to_string(CURRENT_TIME_MILLI.count()), iter_num, list_signs);
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  const schema::RequestAllClientListSign *get_list_sign_req =
    flatbuffers::GetRoot<schema::RequestAllClientListSign>(req_data);
  if (get_list_sign_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for RequestAllClientListSign.";
    BuildGetListSignKernelRsp(fbb, schema::ResponseCode_RequestError, reason,
                              std::to_string(CURRENT_TIME_MILLI.count()), iter_num, list_signs);
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  // verify signature
  if (FLContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(get_list_sign_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed.";
      BuildGetListSignKernelRsp(fbb, schema::ResponseCode_RequestError, reason,
                                std::to_string(CURRENT_TIME_MILLI.count()), iter_num, list_signs);
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed.";
      BuildGetListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, std::to_string(CURRENT_TIME_MILLI.count()),
                                iter_num, list_signs);
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    if (verify_result == sigVerifyResult::PASSED) {
      MS_LOG(INFO) << "verify signature passed!";
    }
  }

  size_t iter_client = IntToSize(get_list_sign_req->iteration());
  if (iter_num != iter_client) {
    MS_LOG(ERROR) << "get list sign iteration number is invalid: server now iteration is " << iter_num
                  << ". client request iteration is " << iter_client;
    BuildGetListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, "iter num is error.",
                              std::to_string(CURRENT_TIME_MILLI.count()), iter_num, list_signs);
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  std::string fl_id = get_list_sign_req->fl_id()->str();
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(WARNING) << "Current amount for GetListSignKernel is enough.";
  }
  if (!GetListSign(iter_num, std::to_string(CURRENT_TIME_MILLI.count()), get_list_sign_req, fbb)) {
    MS_LOG(WARNING) << "get list signs not ready.";
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  std::string count_reason = "";
  if (!DistributedCountService::GetInstance().Count(name_)) {
    std::string reason = "Counting for get list sign request failed. Please retry later. " + count_reason;
    BuildGetListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, std::to_string(CURRENT_TIME_MILLI.count()),
                              iter_num, list_signs);
    MS_LOG(ERROR) << reason;
    return true;
  }
  SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
  return true;
}

bool GetListSignKernel::GetListSign(const size_t cur_iterator, const std::string &next_req_time,
                                    const schema::RequestAllClientListSign *get_list_sign_req,
                                    const std::shared_ptr<fl::FBBuilder> &fbb) {
  MS_LOG(INFO) << "CipherMgr::SendClientListSign START";
  std::map<std::string, std::vector<unsigned char>> client_list_signs_empty;
  std::map<std::string, std::vector<unsigned char>> client_list_signs_all;
  std::unordered_map<std::string, std::string> client_list_sings_strs;
  auto status = cache::ClientInfos::GetInstance().GetAllClientListSign(&client_list_sings_strs);
  if (!status.IsSuccess()) {
    MS_LOG(WARNING) << "Failed to get all client list signatures";
    BuildGetListSignKernelRsp(fbb, schema::ResponseCode_SucNotReady, "The server is not ready.", next_req_time,
                              cur_iterator, client_list_signs_empty);
    return false;
  }
  size_t cur_clients_sign_num = client_list_sings_strs.size();
  if (cur_clients_sign_num < cipher_init_->push_list_sign_threshold) {
    MS_LOG(INFO) << "The server is not ready. push_list_sign_needed: " << cipher_init_->push_list_sign_threshold;
    MS_LOG(INFO) << "now push_sign_client_num: " << cur_clients_sign_num;
    BuildGetListSignKernelRsp(fbb, schema::ResponseCode_SucNotReady, "The server is not ready.", next_req_time,
                              cur_iterator, client_list_signs_empty);
    return false;
  }

  for (auto &item : client_list_sings_strs) {
    std::vector<uint8_t> signature(item.second.begin(), item.second.end());
    (void)client_list_signs_all.emplace(std::pair<std::string, std::vector<uint8_t>>(item.first, signature));
  }

  MS_ERROR_IF_NULL_W_RET_VAL(get_list_sign_req, false);
  MS_ERROR_IF_NULL_W_RET_VAL(get_list_sign_req->fl_id(), false);
  std::string fl_id = get_list_sign_req->fl_id()->str();
  if (client_list_signs_all.find(fl_id) == client_list_signs_all.end()) {
    std::string reason;
    auto has_client = fl::cache::ClientInfos::GetInstance().HasUpdateModelClient(fl_id);
    if (has_client) {
      reason = "client not send list signature, but in update model client list.";
      BuildGetListSignKernelRsp(fbb, schema::ResponseCode_SUCCEED, reason, next_req_time, cur_iterator,
                                client_list_signs_all);
    } else {
      reason = "client not send list signature, && client is illegal";
      BuildGetListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, next_req_time, cur_iterator,
                                client_list_signs_empty);
    }
    MS_LOG(WARNING) << reason;
    return false;
  }
  std::string reason = "send update model client list signature success. ";
  BuildGetListSignKernelRsp(fbb, schema::ResponseCode_SUCCEED, reason, next_req_time, cur_iterator,
                            client_list_signs_all);
  MS_LOG(INFO) << "CipherMgr::Send Client ListSign Success";
  return true;
}

void GetListSignKernel::BuildGetListSignKernelRsp(const std::shared_ptr<FBBuilder> &fbb,
                                                  const schema::ResponseCode retcode, const std::string &reason,
                                                  const std::string &next_req_time, const size_t iteration,
                                                  const std::map<std::string, std::vector<unsigned char>> &list_signs) {
  auto rsp_reason = fbb->CreateString(reason);
  auto rsp_next_req_time = fbb->CreateString(next_req_time);
  if (list_signs.size() == 0) {
    schema::ReturnAllClientListSignBuilder rsp_builder(*(fbb.get()));
    rsp_builder.add_retcode(static_cast<int>(retcode));
    rsp_builder.add_reason(rsp_reason);
    rsp_builder.add_next_req_time(rsp_next_req_time);
    rsp_builder.add_iteration(SizeToInt(iteration));
    auto rsp_get_list_sign = rsp_builder.Finish();
    fbb->Finish(rsp_get_list_sign);
    return;
  }
  std::vector<flatbuffers::Offset<schema::ClientListSign>> client_list_signs;
  for (auto iter = list_signs.begin(); iter != list_signs.end(); ++iter) {
    auto fbs_fl_id = fbb->CreateString(iter->first);
    auto fbs_sign = fbb->CreateVector(iter->second.data(), iter->second.size());
    auto cur_sign = schema::CreateClientListSign(*fbb, fbs_fl_id, fbs_sign);
    client_list_signs.push_back(cur_sign);
  }
  auto all_signs = fbb->CreateVector(client_list_signs);
  schema::ReturnAllClientListSignBuilder rsp_builder(*(fbb.get()));
  rsp_builder.add_retcode(static_cast<int>(retcode));
  rsp_builder.add_reason(rsp_reason);
  rsp_builder.add_next_req_time(rsp_next_req_time);
  rsp_builder.add_iteration(SizeToInt(iteration));
  rsp_builder.add_client_list_sign(all_signs);
  auto rsp_get_list_sign = rsp_builder.Finish();
  fbb->Finish(rsp_get_list_sign);
  return;
}

REG_ROUND_KERNEL(getListSign, GetListSignKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
