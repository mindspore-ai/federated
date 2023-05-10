/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "distributed_cache/client_infos.h"
#include <memory>
#include <string>
#include <vector>
#include "distributed_cache/redis_keys.h"
#include "distributed_cache/timer.h"
#include "common/fl_context.h"
#include "distributed_cache/common.h"

namespace mindspore {
namespace fl {
namespace cache {
std::shared_ptr<RedisClientBase> ClientInfos::GetOneClient() {
  return DistributedCacheLoader::Instance().GetOneClient();
}

CacheStatus ClientInfos::AddPbItem(const std::string &name, const std::string &fl_id,
                                   const google::protobuf::Message &value) {
  auto pb_value = value.SerializeAsString();
  return AddPbItem(name, fl_id, pb_value);
}

CacheStatus ClientInfos::AddPbItemList(const std::string &name, const google::protobuf::Message &value) {
  auto pb_value = value.SerializeAsString();
  return AddPbItemList(name, pb_value);
}

CacheStatus ClientInfos::AddPbItem(const std::string &name, const std::string &fl_id, const std::string &value) {
  auto client = GetOneClient();
  if (client == nullptr) {
    THROW_CACHE_UNAVAILABLE;
  }
  auto ret = client->HSetNx(name, fl_id, value);
  if (ret == kCacheNetErr) {
    THROW_CACHE_UNAVAILABLE;
  }
  if (!ret.IsSuccess()) {
    return ret;
  }
  return client->Expire(name, Timer::iteration_expire_time_in_seconds());
}

CacheStatus ClientInfos::AddPbItemList(const std::string &name, const std::string &value) {
  auto client = GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "THROW_CACHE_UNAVAILABLE";
    THROW_CACHE_UNAVAILABLE;
  }
  auto ret = client->LPush(name, value);
  if (ret == kCacheNetErr) {
    THROW_CACHE_UNAVAILABLE;
    MS_LOG_WARNING << "THROW_CACHE_UNAVAILABLE";
  }
  if (!ret.IsSuccess()) {
    MS_LOG_WARNING << "AddPbItemList ERROR";
    return ret;
  }
  return client->Expire(name, Timer::unsupervised_data_expire_time_in_seconds());
}

bool ClientInfos::HasPbItem(const std::string &name, const std::string &fl_id) {
  auto client = GetOneClient();
  if (client == nullptr) {
    THROW_CACHE_UNAVAILABLE;
  }
  bool value = false;
  auto cache_ret = client->HExists(name, fl_id, &value);
  if (!cache_ret.IsSuccess()) {
    THROW_CACHE_UNAVAILABLE;
  }
  return value;
}

CacheStatus ClientInfos::GetPbItem(const std::string &name, const std::string &fl_id, std::string *value) {
  if (value == nullptr) {
    return kCacheInnerErr;
  }
  auto client = GetOneClient();
  if (client == nullptr) {
    THROW_CACHE_UNAVAILABLE;
  }
  auto ret = client->HGet(name, fl_id, value);
  if (ret == kCacheNetErr) {
    THROW_CACHE_UNAVAILABLE;
  }
  return ret;
}

CacheStatus ClientInfos::GetAllPbItems(const std::string &name, std::unordered_map<std::string, std::string> *items) {
  if (items == nullptr) {
    return kCacheInnerErr;
  }
  auto client = GetOneClient();
  if (client == nullptr) {
    THROW_CACHE_UNAVAILABLE;
  }
  auto ret = client->HGetAll(name, items);
  if (ret == kCacheNetErr) {
    THROW_CACHE_UNAVAILABLE;
  }
  return ret;
}

CacheStatus ClientInfos::GetAllClients(const std::string &name, std::vector<std::string> *items) {
  if (items == nullptr) {
    return kCacheInnerErr;
  }
  auto client = GetOneClient();
  if (client == nullptr) {
    THROW_CACHE_UNAVAILABLE;
  }
  auto ret = client->SMembers(name, items);
  if (ret == kCacheNetErr) {
    THROW_CACHE_UNAVAILABLE;
  }
  return ret;
}

CacheStatus ClientInfos::AddFlItem(const std::string &name, const std::string &fl_id) {
  auto client = GetOneClient();
  if (client == nullptr) {
    THROW_CACHE_UNAVAILABLE;
  }
  auto ret = client->SAdd(name, fl_id);
  if (ret == kCacheNetErr) {
    THROW_CACHE_UNAVAILABLE;
  }
  if (!ret.IsSuccess()) {
    return ret;
  }
  return client->Expire(name, Timer::iteration_expire_time_in_seconds());
}

bool ClientInfos::HasFlItem(const std::string &name, const std::string &fl_id) {
  auto client = GetOneClient();
  if (client == nullptr) {
    THROW_CACHE_UNAVAILABLE;
  }
  bool value = false;
  auto ret = client->SIsMember(name, fl_id, &value);
  if (!ret.IsSuccess()) {
    THROW_CACHE_UNAVAILABLE;
  }
  return value;
}

CacheStatus ClientInfos::SetPbValue(const std::string &name, const std::string &value) {
  auto client = GetOneClient();
  if (client == nullptr) {
    THROW_CACHE_UNAVAILABLE;
  }
  auto ret = client->SetEx(name, value, Timer::iteration_expire_time_in_seconds());
  if (ret == kCacheNetErr) {
    THROW_CACHE_UNAVAILABLE;
  }
  return ret;
}

CacheStatus ClientInfos::GetPbValue(const std::string &name, std::string *value) {
  if (value == nullptr) {
    return kCacheInnerErr;
  }
  auto client = GetOneClient();
  if (client == nullptr) {
    THROW_CACHE_UNAVAILABLE;
  }
  auto ret = client->Get(name, value);
  if (ret == kCacheNetErr) {
    THROW_CACHE_UNAVAILABLE;
  }
  return ret;
}

CacheStatus ClientInfos::SetPbValue(const std::string &name, const google::protobuf::Message &value) {
  auto pb_value = value.SerializeAsString();
  return SetPbValue(name, pb_value);
}

CacheStatus ClientInfos::AddDeviceMeta(const std::string &fl_id, const DeviceMeta &value) {
  auto key = RedisKeys::GetInstance().ClientDeviceMetasHash();
  return AddPbItem(key, fl_id, value);
}

CacheStatus ClientInfos::GetDeviceMeta(const std::string &fl_id, DeviceMeta *value) {
  auto key = RedisKeys::GetInstance().ClientDeviceMetasHash();
  return GetPbItem(key, fl_id, value);
}

bool ClientInfos::HasDeviceMeta(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientDeviceMetasHash();
  return HasPbItem(key, fl_id);
}

CacheStatus ClientInfos::AddClientKeyAttestation(const std::string &fl_id, const std::string &value) {
  auto key = RedisKeys::GetInstance().ClientKeyAttestationHash();
  return AddPbItem(key, fl_id, value);
}

CacheStatus ClientInfos::GetClientKeyAttestation(const std::string &fl_id, std::string *value) {
  auto key = RedisKeys::GetInstance().ClientKeyAttestationHash();
  return GetPbItem(key, fl_id, value);
}

bool ClientInfos::HasClientKeyAttestation(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientKeyAttestationHash();
  return HasPbItem(key, fl_id);
}

CacheStatus ClientInfos::AddClientKey(const std::string &fl_id, const KeysPb &value) {
  auto key = RedisKeys::GetInstance().ClientKeysHash();
  return AddPbItem(key, fl_id, value);
}

CacheStatus ClientInfos::GetClientKey(const std::string &fl_id, KeysPb *value) {
  auto key = RedisKeys::GetInstance().ClientKeysHash();
  return GetPbItem(key, fl_id, value);
}

bool ClientInfos::HasClientKey(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientKeysHash();
  return HasPbItem(key, fl_id);
}

CacheStatus ClientInfos::GetAllClientKeys(std::unordered_map<std::string, KeysPb> *value) {
  auto key = RedisKeys::GetInstance().ClientKeysHash();
  return GetAllPbItems(key, value);
}

CacheStatus ClientInfos::AddClientEncryptedShares(const std::string &fl_id, const SharesPb &value) {
  auto key = RedisKeys::GetInstance().ClientEncryptedSharesHash();
  return AddPbItem(key, fl_id, value);
}

CacheStatus ClientInfos::GetClientEncryptedShare(const std::string &fl_id, SharesPb *value) {
  auto key = RedisKeys::GetInstance().ClientEncryptedSharesHash();
  return GetPbItem(key, fl_id, value);
}

bool ClientInfos::HasClientEncryptedShare(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientEncryptedSharesHash();
  return HasPbItem(key, fl_id);
}

CacheStatus ClientInfos::GetAllClientEncryptedShares(std::unordered_map<std::string, SharesPb> *value) {
  auto key = RedisKeys::GetInstance().ClientEncryptedSharesHash();
  return GetAllPbItems(key, value);
}

CacheStatus ClientInfos::AddClientRestructShares(const std::string &fl_id, const SharesPb &value) {
  auto key = RedisKeys::GetInstance().ClientRestructSharesHash();
  return AddPbItem(key, fl_id, value);
}

CacheStatus ClientInfos::GetClientRestructShare(const std::string &fl_id, SharesPb *value) {
  auto key = RedisKeys::GetInstance().ClientRestructSharesHash();
  return GetPbItem(key, fl_id, value);
}

bool ClientInfos::HasClientRestructShare(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientRestructSharesHash();
  return HasPbItem(key, fl_id);
}

CacheStatus ClientInfos::GetAllClientRestructShares(std::unordered_map<std::string, SharesPb> *value) {
  auto key = RedisKeys::GetInstance().ClientRestructSharesHash();
  return GetAllPbItems(key, value);
}

CacheStatus ClientInfos::AddClientListSign(const std::string &fl_id, const std::string &value) {
  auto key = RedisKeys::GetInstance().ClientSignaturesHash();
  return AddPbItem(key, fl_id, value);
}

CacheStatus ClientInfos::GetClientListSign(const std::string &fl_id, std::string *value) {
  auto key = RedisKeys::GetInstance().ClientSignaturesHash();
  return GetPbItem(key, fl_id, value);
}

bool ClientInfos::HasClientListSign(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientSignaturesHash();
  return HasPbItem(key, fl_id);
}

CacheStatus ClientInfos::GetAllClientListSign(std::unordered_map<std::string, std::string> *value) {
  auto key = RedisKeys::GetInstance().ClientSignaturesHash();
  return GetAllPbItems(key, value);
}

// fl set value
CacheStatus ClientInfos::AddExchangeKeyClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientExchangeKeysFlSet();
  return AddFlItem(key, fl_id);
}

bool ClientInfos::HasExchangeKeyClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientExchangeKeysFlSet();
  return HasFlItem(key, fl_id);
}

CacheStatus ClientInfos::GetAllExchangeKeyClients(std::vector<std::string> *value) {
  auto key = RedisKeys::GetInstance().ClientExchangeKeysFlSet();
  return GetAllClients(key, value);
}

CacheStatus ClientInfos::AddGetKeysClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientGetKeysFlSet();
  return AddFlItem(key, fl_id);
}

bool ClientInfos::HasGetKeysClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientGetKeysFlSet();
  return HasFlItem(key, fl_id);
}

CacheStatus ClientInfos::AddShareSecretsClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientShareSecretsFlSet();
  return AddFlItem(key, fl_id);
}

bool ClientInfos::HasShareSecretsClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientShareSecretsFlSet();
  return HasFlItem(key, fl_id);
}

CacheStatus ClientInfos::GetAllShareSecretsClients(std::vector<std::string> *value) {
  auto key = RedisKeys::GetInstance().ClientShareSecretsFlSet();
  return GetAllClients(key, value);
}

CacheStatus ClientInfos::AddGetSecretsClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientGetSecretsFlSet();
  return AddFlItem(key, fl_id);
}

bool ClientInfos::HasGetSecretsClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientGetSecretsFlSet();
  return HasFlItem(key, fl_id);
}

CacheStatus ClientInfos::AddUpdateModelClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientUpdateModelFlSet();
  return AddFlItem(key, fl_id);
}

bool ClientInfos::HasUpdateModelClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientUpdateModelFlSet();
  return HasFlItem(key, fl_id);
}

CacheStatus ClientInfos::GetAllUpdateModelClients(std::vector<std::string> *value) {
  auto key = RedisKeys::GetInstance().ClientUpdateModelFlSet();
  return GetAllClients(key, value);
}

CacheStatus ClientInfos::AddGetUpdateModelClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientGetUpdateModelFlSet();
  return AddFlItem(key, fl_id);
}

bool ClientInfos::HasGetUpdateModelClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientGetUpdateModelFlSet();
  return HasFlItem(key, fl_id);
}

CacheStatus ClientInfos::AddReconstructClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientReconstructFlSet();
  return AddFlItem(key, fl_id);
}

bool ClientInfos::HasReconstructClient(const std::string &fl_id) {
  auto key = RedisKeys::GetInstance().ClientReconstructFlSet();
  return HasFlItem(key, fl_id);
}

CacheStatus ClientInfos::SetClientNoises(const ClientNoises &noises) {
  auto key = RedisKeys::GetInstance().ClientNoisesString();
  return SetPbValue(key, noises);
}

CacheStatus ClientInfos::GetClientNoises(ClientNoises *noises) {
  auto key = RedisKeys::GetInstance().ClientNoisesString();
  return GetPbValue(key, noises);
}

CacheStatus ClientInfos::AddUnsupervisedEvalItem(const UnsupervisedEvalItem &unsupervised_eval_item) {
  auto key = RedisKeys::GetInstance().ClientUnsupervisedEvalHash();
  return AddPbItemList(key, unsupervised_eval_item);
}

CacheStatus ClientInfos::AddSignDSbHat(const std::string &b_hat) {
  auto key = RedisKeys::GetInstance().ClientSignDSbHatHash();
  return AddPbItemList(key, b_hat);
}

bool ClientInfos::ResetOnNewIteration() {
  std::vector<std::string> del_keys = {
    RedisKeys::GetInstance().ClientDeviceMetasHash(),    RedisKeys::GetInstance().ClientKeyAttestationHash(),
    RedisKeys::GetInstance().ClientKeysHash(),           RedisKeys::GetInstance().ClientEncryptedSharesHash(),
    RedisKeys::GetInstance().ClientRestructSharesHash(), RedisKeys::GetInstance().ClientSignaturesHash(),
    RedisKeys::GetInstance().ClientExchangeKeysFlSet(),  RedisKeys::GetInstance().ClientGetKeysFlSet(),
    RedisKeys::GetInstance().ClientShareSecretsFlSet(),  RedisKeys::GetInstance().ClientGetSecretsFlSet(),
    RedisKeys::GetInstance().ClientUpdateModelFlSet(),   RedisKeys::GetInstance().ClientGetUpdateModelFlSet(),
    RedisKeys::GetInstance().ClientReconstructFlSet()};
  auto client = GetOneClient();
  if (client == nullptr) {
    MS_LOG_ERROR << "Get redis client failed";
    return false;
  }
  constexpr uint64_t rel_time_in_seconds = 60;  // 60sec release time
  for (auto &item : del_keys) {
    client->Expire(item, rel_time_in_seconds);
  }
  return true;
}
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
