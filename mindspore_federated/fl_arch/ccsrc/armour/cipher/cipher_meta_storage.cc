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

#include "armour/cipher/cipher_meta_storage.h"
#include <unordered_map>
#include "distributed_cache/client_infos.h"
#include "distributed_cache/instance_context.h"

namespace mindspore {
namespace fl {
namespace armour {
void CipherMetaStorage::GetClientSharesFromServerInner(
  const std::unordered_map<std::string, fl::SharesPb> &value_map,
  std::map<std::string, std::vector<clientshare_str>> *clients_shares_list) {
  if (clients_shares_list == nullptr) {
    MS_LOG(ERROR) << "input clients_shares_list is nullptr";
    return;
  }
  for (auto &item : value_map) {
    auto &fl_id = item.first;
    const fl::SharesPb &shares_pb = item.second;
    std::vector<clientshare_str> &encrpted_shares_new = (*clients_shares_list)[fl_id];
    size_t client_share_num = IntToSize(shares_pb.clientsharestrs_size());
    for (size_t index_shares = 0; index_shares < client_share_num; ++index_shares) {
      const fl::ClientShareStr &client_share_str_pb = shares_pb.clientsharestrs(index_shares);
      clientshare_str new_clientshare;
      new_clientshare.fl_id = client_share_str_pb.fl_id();
      new_clientshare.index = client_share_str_pb.index();
      new_clientshare.share.assign(client_share_str_pb.share().begin(), client_share_str_pb.share().end());
      encrpted_shares_new.push_back(new_clientshare);
    }
  }
}

void CipherMetaStorage::GetClientReconstructSharesFromServer(
  std::map<std::string, std::vector<clientshare_str>> *clients_shares_list) {
  if (clients_shares_list == nullptr) {
    MS_LOG(ERROR) << "input clients_shares_list is nullptr";
    return;
  }
  fl::cache::CacheStatus status;
  std::unordered_map<std::string, fl::SharesPb> value_map;
  status = fl::cache::ClientInfos::GetInstance().GetAllClientRestructShares(&value_map);
  if (!status.IsSuccess()) {
    MS_LOG_WARNING << "Get ClientRestructShares from cache failed";
    return;
  }
  GetClientSharesFromServerInner(value_map, clients_shares_list);
}

void CipherMetaStorage::GetClientEncryptedSharesFromServer(
  std::map<std::string, std::vector<clientshare_str>> *clients_shares_list) {
  if (clients_shares_list == nullptr) {
    MS_LOG(ERROR) << "input clients_shares_list is nullptr";
    return;
  }
  fl::cache::CacheStatus status;
  std::unordered_map<std::string, fl::SharesPb> value_map;
  status = fl::cache::ClientInfos::GetInstance().GetAllClientEncryptedShares(&value_map);
  if (!status.IsSuccess()) {
    MS_LOG_WARNING << "Get ClientRestructShares from cache failed";
    return;
  }
  GetClientSharesFromServerInner(value_map, clients_shares_list);
}

void CipherMetaStorage::GetClientKeysFromServer(
  std::map<std::string, std::vector<std::vector<uint8_t>>> *clients_keys_list) {
  if (clients_keys_list == nullptr) {
    MS_LOG(ERROR) << "input clients_keys_list is nullptr";
    return;
  }
  std::unordered_map<std::string, fl::KeysPb> keys;
  auto ret = fl::cache::ClientInfos::GetInstance().GetAllClientKeys(&keys);
  if (!ret.IsSuccess()) {
    MS_LOG(ERROR) << "Get client keys from cache failed";
    return;
  }
  for (auto &item : keys) {
    auto &fl_id = item.first;
    auto &keys_pb = item.second;
    auto &key0 = keys_pb.key(0);
    auto &key1 = keys_pb.key(1);
    std::vector<uint8_t> cpk(key0.begin(), key0.end());
    std::vector<uint8_t> spk(key1.begin(), key1.end());
    (void)clients_keys_list->emplace(std::pair<std::string, std::vector<std::vector<uint8_t>>>(fl_id, {cpk, spk}));
  }
}

void CipherMetaStorage::GetStableClientKeysFromServer(
  std::map<std::string, std::vector<std::vector<uint8_t>>> *clients_keys_list) {
  if (clients_keys_list == nullptr) {
    MS_LOG(ERROR) << "Input clients_keys_list is nullptr";
    return;
  }
  std::unordered_map<std::string, fl::KeysPb> keys;
  auto ret = fl::cache::ClientInfos::GetInstance().GetAllClientKeys(&keys);
  if (!ret.IsSuccess()) {
    MS_LOG(ERROR) << "Get client keys from cache failed";
    return;
  }
  for (auto &item : keys) {
    auto &fl_id = item.first;
    auto &keys_pb = item.second;
    auto &key0 = keys_pb.key(0);
    std::vector<uint8_t> spk(key0.begin(), key0.end());
    (void)clients_keys_list->emplace(std::pair<std::string, std::vector<std::vector<uint8_t>>>(fl_id, {spk}));
  }
}

void CipherMetaStorage::GetClientIVsFromServer(
  std::map<std::string, std::vector<std::vector<uint8_t>>> *clients_ivs_list) {
  if (clients_ivs_list == nullptr) {
    MS_LOG(ERROR) << "input clients_ivs_list is nullptr";
    return;
  }
  std::unordered_map<std::string, fl::KeysPb> keys;
  auto ret = fl::cache::ClientInfos::GetInstance().GetAllClientKeys(&keys);
  if (!ret.IsSuccess()) {
    MS_LOG(ERROR) << "Get client keys from cache failed";
    return;
  }
  for (auto &item : keys) {
    auto &fl_id = item.first;
    auto &keys_pb = item.second;
    std::vector<uint8_t> ind_iv(keys_pb.ind_iv().begin(), keys_pb.ind_iv().end());
    std::vector<uint8_t> pw_iv(keys_pb.pw_iv().begin(), keys_pb.pw_iv().end());
    std::vector<uint8_t> pw_salt(keys_pb.pw_salt().begin(), keys_pb.pw_salt().end());

    std::vector<std::vector<uint8_t>> cur_ivs;
    cur_ivs.push_back(ind_iv);
    cur_ivs.push_back(pw_iv);
    cur_ivs.push_back(pw_salt);
    (void)clients_ivs_list->emplace(std::pair<std::string, std::vector<std::vector<uint8_t>>>(fl_id, cur_ivs));
  }
}

bool CipherMetaStorage::GetClientNoisesFromServer(std::vector<float> *cur_public_noise) {
  if (cur_public_noise == nullptr) {
    MS_LOG(ERROR) << "input cur_public_noise is nullptr";
    return false;
  }
  fl::ClientNoises noises;
  constexpr int count_threshold = 1000;
  constexpr int register_time = 500;
  for (int count = 0; count < count_threshold; count++) {
    (void)fl::cache::ClientInfos::GetInstance().GetClientNoises(&noises);
    if (noises.has_one_client_noises()) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(register_time));
  }
  if (!noises.has_one_client_noises()) {
    MS_LOG(WARNING) << "GetClientNoisesFromServer failed";
    return false;
  }
  auto &noises_data = noises.one_client_noises().noise();
  cur_public_noise->assign(noises_data.begin(), noises_data.end());
  return true;
}

bool CipherMetaStorage::GetPrimeFromServer(uint8_t *prime) {
  if (prime == nullptr) {
    MS_LOG(ERROR) << "input prime is nullptr";
    return false;
  }
  auto prime_str = fl::cache::InstanceContext::Instance().GetPrime();
  if (prime_str.size() != PRIME_MAX_LEN) {
    MS_LOG(ERROR) << "get prime size is :" << prime_str.size();
    return false;
  } else {
    if (memcpy_s(prime, PRIME_MAX_LEN, prime_str.data(), prime_str.size()) != 0) {
      MS_LOG(ERROR) << "Memcpy_s error";
      return false;
    }
    return true;
  }
}

void CipherMetaStorage::RegisterPrime(const std::string &prime) {
  fl::cache::InstanceContext::Instance().SetPrime(prime);
}

bool CipherMetaStorage::UpdateClientKeyToServer(const schema::RequestExchangeKeys *exchange_keys_req) {
  std::string fl_id = exchange_keys_req->fl_id()->str();
  auto fbs_cpk = exchange_keys_req->c_pk();
  auto fbs_spk = exchange_keys_req->s_pk();
  if (fbs_cpk == nullptr || fbs_spk == nullptr) {
    MS_LOG(ERROR) << "public key from exchange_keys_req is null";
    return false;
  }

  size_t spk_len = fbs_spk->size();
  size_t cpk_len = fbs_cpk->size();

  // transform fbs (fbs_cpk & fbs_spk) to a vector: public_key
  std::vector<std::vector<uint8_t>> cur_public_key;
  std::vector<uint8_t> cpk(cpk_len);
  std::vector<uint8_t> spk(spk_len);
  bool ret_create_code_cpk = CreateArray<uint8_t>(&cpk, *fbs_cpk);
  bool ret_create_code_spk = CreateArray<uint8_t>(&spk, *fbs_spk);
  if (!(ret_create_code_cpk && ret_create_code_spk)) {
    MS_LOG(ERROR) << "create array for public keys failed";
    return false;
  }
  cur_public_key.push_back(cpk);
  cur_public_key.push_back(spk);

  auto fbs_signature = exchange_keys_req->signature();
  std::vector<char> signature;
  if (fbs_signature == nullptr) {
    MS_LOG(WARNING) << "signature in exchange_keys_req is nullptr";
  } else {
    signature.assign(fbs_signature->begin(), fbs_signature->end());
  }

  auto fbs_ind_iv = exchange_keys_req->ind_iv();
  std::vector<char> ind_iv;
  if (fbs_ind_iv == nullptr) {
    MS_LOG(WARNING) << "ind_iv in exchange_keys_req is nullptr";
  } else {
    ind_iv.assign(fbs_ind_iv->begin(), fbs_ind_iv->end());
  }

  auto fbs_pw_iv = exchange_keys_req->pw_iv();
  std::vector<char> pw_iv;
  if (fbs_pw_iv == nullptr) {
    MS_LOG(WARNING) << "pw_iv in exchange_keys_req is nullptr";
  } else {
    pw_iv.assign(fbs_pw_iv->begin(), fbs_pw_iv->end());
  }

  auto fbs_pw_salt = exchange_keys_req->pw_salt();
  std::vector<char> pw_salt;
  if (fbs_pw_salt == nullptr) {
    MS_LOG(WARNING) << "pw_salt in exchange_keys_req is nullptr";
  } else {
    pw_salt.assign(fbs_pw_salt->begin(), fbs_pw_salt->end());
  }

  auto fbs_cert_chain = exchange_keys_req->certificate_chain();
  std::vector<std::string> cert_chain;
  if (fbs_cert_chain == nullptr) {
    MS_LOG(WARNING) << "certificate_chain in exchange_keys_req is nullptr";
  } else {
    for (auto iter = fbs_cert_chain->begin(); iter != fbs_cert_chain->end(); ++iter) {
      cert_chain.push_back(iter->str());
    }
  }

  // update new item to memory server.
  fl::KeysPb keys;
  keys.add_key()->assign(cur_public_key[0].begin(), cur_public_key[0].end());
  keys.add_key()->assign(cur_public_key[1].begin(), cur_public_key[1].end());
  auto timestamp_ptr = exchange_keys_req->timestamp();
  MS_EXCEPTION_IF_NULL(timestamp_ptr);
  keys.set_timestamp(timestamp_ptr->str());
  keys.set_iter_num(exchange_keys_req->iteration());
  keys.set_ind_iv(ind_iv.data(), ind_iv.size());
  keys.set_pw_iv(pw_iv.data(), pw_iv.size());
  keys.set_pw_salt(pw_salt.data(), pw_salt.size());
  keys.set_signature(signature.data(), signature.size());
  for (size_t i = 0; i < cert_chain.size(); i++) {
    keys.add_certificate_chain(cert_chain[i]);
  }
  auto ret = fl::cache::ClientInfos::GetInstance().AddClientKey(fl_id, keys);
  return ret.IsSuccess();
}

bool CipherMetaStorage::UpdateStableClientKeyToServer(const schema::RequestExchangeKeys *exchange_keys_req) {
  auto fbs_fl_id = exchange_keys_req->fl_id();
  MS_EXCEPTION_IF_NULL(fbs_fl_id);
  std::string fl_id = fbs_fl_id->str();
  auto fbs_spk = exchange_keys_req->s_pk();
  if (fbs_spk == nullptr) {
    MS_LOG(ERROR) << "Public key from exchange_keys_req is null";
    return false;
  }

  size_t spk_len = fbs_spk->size();

  // transform fbs_spk to a vector: public_key
  std::vector<uint8_t> spk(spk_len);
  bool ret_create_code_spk = CreateArray<uint8_t>(&spk, *fbs_spk);
  if (!ret_create_code_spk) {
    MS_LOG(ERROR) << "Create array for public keys failed";
    return false;
  }

  auto fbs_pw_iv = exchange_keys_req->pw_iv();
  auto fbs_pw_salt = exchange_keys_req->pw_salt();
  std::vector<char> pw_iv;
  std::vector<char> pw_salt;
  if (fbs_pw_iv == nullptr) {
    MS_LOG(WARNING) << "pw_iv in exchange_keys_req is nullptr";
  } else {
    pw_iv.assign(fbs_pw_iv->begin(), fbs_pw_iv->end());
  }
  if (fbs_pw_salt == nullptr) {
    MS_LOG(WARNING) << "pw_salt in exchange_keys_req is nullptr";
  } else {
    pw_salt.assign(fbs_pw_salt->begin(), fbs_pw_salt->end());
  }

  // update new item to memory server.
  fl::KeysPb keys;
  (void)keys.add_key()->assign(spk.begin(), spk.end());
  keys.set_pw_iv(pw_iv.data(), pw_iv.size());
  keys.set_pw_salt(pw_salt.data(), pw_salt.size());
  auto ret = fl::cache::ClientInfos::GetInstance().AddClientKey(fl_id, keys);
  return ret.IsSuccess();
}

bool CipherMetaStorage::UpdateClientNoiseToServer(const std::vector<float> &cur_public_noise) {
  // update new item to memory server.
  fl::ClientNoises noises_pb;
  *(noises_pb.mutable_one_client_noises()->mutable_noise()) = {cur_public_noise.begin(), cur_public_noise.end()};
  auto ret = fl::cache::ClientInfos::GetInstance().SetClientNoises(noises_pb);
  return ret.IsSuccess();
}

bool CipherMetaStorage::UpdateClientReconstructShareToServer(
  const std::string &fl_id, const flatbuffers::Vector<flatbuffers::Offset<schema::ClientShare>> *shares) {
  fl::SharesPb shares_pb;
  auto ret = UpdateClientShareToServerInner(fl_id, shares, &shares_pb);
  if (!ret) {
    return false;
  }
  auto status = fl::cache::ClientInfos::GetInstance().AddClientRestructShares(fl_id, shares_pb);
  return status.IsSuccess();
}

bool CipherMetaStorage::UpdateClientEncryptedShareToServer(
  const std::string &fl_id, const flatbuffers::Vector<flatbuffers::Offset<schema::ClientShare>> *shares) {
  fl::SharesPb shares_pb;
  auto ret = UpdateClientShareToServerInner(fl_id, shares, &shares_pb);
  if (!ret) {
    return false;
  }
  auto status = fl::cache::ClientInfos::GetInstance().AddClientEncryptedShares(fl_id, shares_pb);
  return status.IsSuccess();
}

bool CipherMetaStorage::UpdateClientShareToServerInner(
  const std::string &fl_id, const flatbuffers::Vector<flatbuffers::Offset<schema::ClientShare>> *shares,
  fl::SharesPb *shares_pb) {
  if (shares == nullptr || shares_pb == nullptr) {
    return false;
  }
  size_t size_shares = shares->size();
  for (size_t index = 0; index < size_shares; ++index) {
    // new item
    fl::ClientShareStr *client_share_str_new_p = shares_pb->add_clientsharestrs();
    std::string fl_id_new = (*shares)[SizeToInt(index)]->fl_id()->str();
    int index_new = (*shares)[SizeToInt(index)]->index();
    auto share = (*shares)[SizeToInt(index)]->share();
    if (share == nullptr) return false;
    client_share_str_new_p->set_share(reinterpret_cast<const char *>(share->data()), share->size());
    client_share_str_new_p->set_fl_id(fl_id_new);
    client_share_str_new_p->set_index(index_new);
  }
  return true;
}
}  // namespace armour
}  // namespace fl
}  // namespace mindspore
