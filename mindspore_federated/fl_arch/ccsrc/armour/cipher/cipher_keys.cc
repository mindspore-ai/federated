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

#include "armour/cipher/cipher_keys.h"
#include <unordered_map>
#include "distributed_cache/client_infos.h"
#include "distributed_cache/instance_context.h"
#include "server/distributed_count_service.h"

namespace mindspore {
namespace fl {
namespace armour {
bool CipherKeys::GetKeys(const size_t cur_iterator, const std::string &next_req_time,
                         const schema::GetExchangeKeys *get_exchange_keys_req, const std::shared_ptr<FBBuilder> &fbb) {
  MS_LOG(INFO) << "CipherMgr::GetKeys START";
  if (get_exchange_keys_req == nullptr) {
    MS_LOG(ERROR) << "Request is nullptr";
    BuildGetKeysRsp(fbb, schema::ResponseCode_RequestError, cur_iterator, next_req_time, false);
    return false;
  }
  if (cipher_init_ == nullptr) {
    BuildGetKeysRsp(fbb, schema::ResponseCode_SystemError, cur_iterator, next_req_time, false);
    return false;
  }
  // get clientlist from memory server.
  std::string encrypt_type = FLContext::instance()->encrypt_type();
  std::string fl_id = get_exchange_keys_req->fl_id()->str();

  bool exchangeKeysOK = fl::server::DistributedCountService::GetInstance().CountReachThreshold("exchangeKeys");
  if (!exchangeKeysOK) {
    MS_LOG(INFO) << "The server is not ready yet: cur_exchangekey_clients_num < exchange_key_threshold";
    BuildGetKeysRsp(fbb, schema::ResponseCode_SucNotReady, cur_iterator, next_req_time, false);
    return false;
  }

  auto found = fl::cache::ClientInfos::GetInstance().HasClientKey(fl_id);
  if (!found) {
    MS_LOG(INFO) << "Get keys: the fl_id: " << fl_id << "is not in exchange keys clients.";
    BuildGetKeysRsp(fbb, schema::ResponseCode_RequestError, cur_iterator, next_req_time, false);
    return false;
  }

  auto ret = fl::cache::ClientInfos::GetInstance().AddGetKeysClient(fl_id);
  if (!ret.IsSuccess()) {
    MS_LOG(ERROR) << "Update get keys clients failed";
    BuildGetKeysRsp(fbb, schema::ResponseCode_OutOfTime, cur_iterator, next_req_time, false);
    return false;
  }

  MS_LOG(INFO) << "GetKeys client list: ";
  if (encrypt_type == kPWEncryptType && FLContext::instance()->pki_verify()) {
    MS_LOG(INFO) << "Build get_keys response in pki_verify mode.";
    BuildPkiVerifyGetKeysRsp(fbb, schema::ResponseCode_SUCCEED, cur_iterator, next_req_time, true);
  } else {
    BuildGetKeysRsp(fbb, schema::ResponseCode_SUCCEED, cur_iterator, next_req_time, true);
  }
  return true;
}

bool CipherKeys::ExchangeKeys(const size_t cur_iterator, const std::string &next_req_time,
                              const schema::RequestExchangeKeys *exchange_keys_req,
                              const std::shared_ptr<FBBuilder> &fbb) {
  MS_LOG(INFO) << "CipherMgr::ExchangeKeys START";
  // step 0: judge if the input param is legal.
  if (exchange_keys_req == nullptr) {
    std::string reason = "Request is nullptr";
    MS_LOG(ERROR) << reason;
    BuildExchangeKeysRsp(fbb, schema::ResponseCode_RequestError, reason, next_req_time, cur_iterator);
    return false;
  }
  if (cipher_init_ == nullptr) {
    std::string reason = "cipher_init_ is nullptr";
    BuildExchangeKeysRsp(fbb, schema::ResponseCode_SystemError, reason, next_req_time, cur_iterator);
    return false;
  }
  std::string fl_id = exchange_keys_req->fl_id()->str();
  fl::DeviceMeta device_meta;
  auto status = fl::cache::ClientInfos::GetInstance().GetDeviceMeta(fl_id, &device_meta);
  MS_LOG(INFO) << "exchange key for fl id " << fl_id;
  if (!status.IsSuccess()) {
    std::string reason = "devices_meta for " + fl_id + " is not set. Please retry later.";
    BuildExchangeKeysRsp(fbb, schema::ResponseCode_OutOfTime, reason, next_req_time, cur_iterator);
    MS_LOG(ERROR) << reason;
    return false;
  }

  // step 1: get clientlist and client keys from memory server.
  std::string encrypt_type = FLContext::instance()->encrypt_type();

  // step2: process new item data. and update new item data to memory server.
  auto found = fl::cache::ClientInfos::GetInstance().HasClientKey(fl_id);
  if (found) {  // the client already exists, return false.
    MS_LOG(ERROR) << "The server has received the request, please do not request again.";
    BuildExchangeKeysRsp(fbb, schema::ResponseCode_SUCCEED,
                         "The server has received the request, please do not request again.", next_req_time,
                         cur_iterator);
    return false;
  }

  bool retcode_key;
  if (encrypt_type == kPWEncryptType) {
    retcode_key = cipher_init_->cipher_meta_storage_.UpdateClientKeyToServer(exchange_keys_req);
  } else {
    retcode_key = cipher_init_->cipher_meta_storage_.UpdateStableClientKeyToServer(exchange_keys_req);
  }

  if (retcode_key) {
    MS_LOG(INFO) << "The client " << fl_id << " CipherMgr::ExchangeKeys Success";
    BuildExchangeKeysRsp(fbb, schema::ResponseCode_SUCCEED, "Success, but the server is not ready yet.", next_req_time,
                         cur_iterator);
    return true;
  } else {
    MS_LOG(ERROR) << "update key or client failed";
    BuildExchangeKeysRsp(fbb, schema::ResponseCode_OutOfTime, "update key or client failed", next_req_time,
                         cur_iterator);
    return false;
  }
}

void CipherKeys::BuildExchangeKeysRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                      const std::string &reason, const std::string &next_req_time,
                                      const size_t iteration) {
  auto rsp_reason = fbb->CreateString(reason);
  auto rsp_next_req_time = fbb->CreateString(next_req_time);

  schema::ResponseExchangeKeysBuilder rsp_builder(*(fbb.get()));
  rsp_builder.add_retcode(retcode);
  rsp_builder.add_reason(rsp_reason);
  rsp_builder.add_next_req_time(rsp_next_req_time);
  rsp_builder.add_iteration(SizeToInt(iteration));
  auto rsp_exchange_keys = rsp_builder.Finish();
  fbb->Finish(rsp_exchange_keys);
  return;
}

void CipherKeys::BuildGetKeysRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                 const size_t iteration, const std::string &next_req_time, bool is_good) {
  if (!is_good) {
    auto fbs_next_req_time = fbb->CreateString(next_req_time);
    schema::ReturnExchangeKeysBuilder rsp_builder(*(fbb.get()));
    rsp_builder.add_retcode(static_cast<int>(retcode));
    rsp_builder.add_iteration(SizeToInt(iteration));
    rsp_builder.add_next_req_time(fbs_next_req_time);
    auto rsp_get_keys = rsp_builder.Finish();
    fbb->Finish(rsp_get_keys);
    return;
  }

  std::unordered_map<std::string, fl::KeysPb> value_map;
  auto status = fl::cache::ClientInfos::GetInstance().GetAllClientKeys(&value_map);
  if (!status.IsSuccess()) {
    MS_LOG(ERROR) << "Get keys from cache failed. Please retry later.";
    BuildGetKeysRsp(fbb, schema::ResponseCode_OutOfTime, iteration, next_req_time, false);
    return;
  }
  std::vector<flatbuffers::Offset<schema::ClientPublicKeys>> public_keys_list;
  for (auto &item : value_map) {
    auto &fl_id = item.first;
    auto &keys_pb = item.second;
    auto fbs_fl_id = fbb->CreateString(fl_id);
    std::vector<uint8_t> pw_iv(keys_pb.pw_iv().begin(), keys_pb.pw_iv().end());
    auto fbs_pw_iv = fbb->CreateVector(pw_iv.data(), pw_iv.size());
    std::vector<uint8_t> pw_salt(keys_pb.pw_salt().begin(), keys_pb.pw_salt().end());
    auto fbs_pw_salt = fbb->CreateVector(pw_salt.data(), pw_salt.size());

    std::string encrypt_type = FLContext::instance()->encrypt_type();
    if (encrypt_type == kPWEncryptType) {
      std::vector<uint8_t> cpk(keys_pb.key(0).begin(), keys_pb.key(0).end());
      std::vector<uint8_t> spk(keys_pb.key(1).begin(), keys_pb.key(1).end());
      auto fbs_c_pk = fbb->CreateVector(cpk.data(), cpk.size());
      auto fbs_s_pk = fbb->CreateVector(spk.data(), spk.size());
      auto cur_public_key = schema::CreateClientPublicKeys(*fbb, fbs_fl_id, fbs_c_pk, fbs_s_pk, fbs_pw_iv, fbs_pw_salt);
      public_keys_list.push_back(cur_public_key);
    } else {
      std::vector<uint8_t> spk(keys_pb.key(0).begin(), keys_pb.key(0).end());
      auto fbs_s_pk = fbb->CreateVector(spk.data(), spk.size());
      auto cur_public_key = schema::CreateClientPublicKeys(*fbb, fbs_fl_id, 0, fbs_s_pk, fbs_pw_iv, fbs_pw_salt);
      public_keys_list.push_back(cur_public_key);
    }
  }

  auto remote_publickeys = fbb->CreateVector(public_keys_list);
  auto fbs_next_req_time = fbb->CreateString(next_req_time);
  schema::ReturnExchangeKeysBuilder rsp_builder(*(fbb.get()));
  rsp_builder.add_retcode(static_cast<int>(retcode));
  rsp_builder.add_iteration(SizeToInt(iteration));
  rsp_builder.add_remote_publickeys(remote_publickeys);
  rsp_builder.add_next_req_time(fbs_next_req_time);
  auto rsp_get_keys = rsp_builder.Finish();
  fbb->Finish(rsp_get_keys);
  MS_LOG(INFO) << "CipherMgr::GetKeys Success";
  return;
}

void CipherKeys::BuildPkiVerifyGetKeysRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                          const size_t iteration, const std::string &next_req_time, bool is_good) {
  if (!is_good) {
    auto fbs_next_req_time = fbb->CreateString(next_req_time);
    schema::ReturnExchangeKeysBuilder rsp_buider(*(fbb.get()));
    rsp_buider.add_retcode(static_cast<int>(retcode));
    rsp_buider.add_iteration(SizeToInt(iteration));
    rsp_buider.add_next_req_time(fbs_next_req_time);
    auto rsp_get_keys = rsp_buider.Finish();
    fbb->Finish(rsp_get_keys);
    return;
  }
  std::unordered_map<std::string, fl::KeysPb> value_map;
  auto status = fl::cache::ClientInfos::GetInstance().GetAllClientKeys(&value_map);
  if (!status.IsSuccess()) {
    MS_LOG(ERROR) << "Get keys from cache failed. Please retry later.";
    BuildGetKeysRsp(fbb, schema::ResponseCode_OutOfTime, iteration, next_req_time, false);
    return;
  }
  std::vector<flatbuffers::Offset<schema::ClientPublicKeys>> public_keys_list;
  for (auto &item : value_map) {
    auto &fl_id = item.first;
    auto &keys_pb = item.second;
    auto fbs_fl_id = fbb->CreateString(fl_id);

    // package public keys for pairwise encryption
    std::vector<uint8_t> cpk(keys_pb.key(0).begin(), keys_pb.key(0).end());
    std::vector<uint8_t> spk(keys_pb.key(1).begin(), keys_pb.key(1).end());
    auto fbs_c_pk = fbb->CreateVector(cpk.data(), cpk.size());
    auto fbs_s_pk = fbb->CreateVector(spk.data(), spk.size());

    std::string timestamp = keys_pb.timestamp();
    auto fbs_timestamp = fbb->CreateString(timestamp);
    int iter_num = keys_pb.iter_num();

    // package initialization vector and salt value for pairwise encryption
    std::vector<uint8_t> pw_iv(keys_pb.pw_iv().begin(), keys_pb.pw_iv().end());
    auto fbs_pw_iv = fbb->CreateVector(pw_iv.data(), pw_iv.size());
    std::vector<uint8_t> pw_salt(keys_pb.pw_salt().begin(), keys_pb.pw_salt().end());
    auto fbs_pw_salt = fbb->CreateVector(pw_salt.data(), pw_salt.size());

    // package signature and certifications
    std::vector<uint8_t> signature(keys_pb.signature().begin(), keys_pb.signature().end());
    auto fbs_sign = fbb->CreateVector(signature.data(), signature.size());
    std::vector<flatbuffers::Offset<flatbuffers::String>> cert_chain;
    for (auto &cert_iter : keys_pb.certificate_chain()) {
      auto fbs_cert = fbb->CreateString(cert_iter);
      cert_chain.push_back(fbs_cert);
    }
    auto fbs_cert_chain = fbb->CreateVector(cert_chain);
    auto cur_public_key = schema::CreateClientPublicKeys(*fbb, fbs_fl_id, fbs_c_pk, fbs_s_pk, fbs_pw_iv, fbs_pw_salt,
                                                         fbs_timestamp, iter_num, fbs_sign, fbs_cert_chain);
    public_keys_list.push_back(cur_public_key);
  }
  auto remote_publickeys = fbb->CreateVector(public_keys_list);
  auto fbs_next_req_time = fbb->CreateString(next_req_time);
  schema::ReturnExchangeKeysBuilder rsp_buider(*(fbb.get()));
  rsp_buider.add_retcode(static_cast<int>(retcode));
  rsp_buider.add_iteration(SizeToInt(iteration));
  rsp_buider.add_remote_publickeys(remote_publickeys);
  rsp_buider.add_next_req_time(fbs_next_req_time);
  auto rsp_get_keys = rsp_buider.Finish();
  fbb->Finish(rsp_get_keys);
  MS_LOG(INFO) << "CipherMgr::GetKeys Success";
  return;
}
}  // namespace armour
}  // namespace fl
}  // namespace mindspore
