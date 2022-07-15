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

#ifndef MINDSPORE_CCSRC_FL_SERVER_CLIENT_INFOS_H_
#define MINDSPORE_CCSRC_FL_SERVER_CLIENT_INFOS_H_

#include <string>
#include <memory>
#include <unordered_map>
#include <set>
#include <vector>
#include <utility>
#include "common/protos/fl.pb.h"
#include "distributed_cache/cache_status.h"
#include "common/common.h"
#include "distributed_cache/distributed_cache.h"

namespace mindspore {
namespace fl {
namespace cache {
class ClientInfos {
 public:
  static ClientInfos &GetInstance() {
    static ClientInfos instance;
    return instance;
  }
  // hash value
  CacheStatus AddDeviceMeta(const std::string &fl_id, const DeviceMeta &value);
  CacheStatus GetDeviceMeta(const std::string &fl_id, DeviceMeta *value);
  bool HasDeviceMeta(const std::string &fl_id);

  CacheStatus AddClientKeyAttestation(const std::string &fl_id, const std::string &value);
  CacheStatus GetClientKeyAttestation(const std::string &fl_id, std::string *value);
  bool HasClientKeyAttestation(const std::string &fl_id);

  CacheStatus AddClientKey(const std::string &fl_id, const KeysPb &value);
  CacheStatus GetClientKey(const std::string &fl_id, KeysPb *value);
  bool HasClientKey(const std::string &fl_id);
  CacheStatus GetAllClientKeys(std::unordered_map<std::string, KeysPb> *value);

  CacheStatus AddClientEncryptedShares(const std::string &fl_id, const SharesPb &value);
  CacheStatus GetClientEncryptedShare(const std::string &fl_id, SharesPb *value);
  bool HasClientEncryptedShare(const std::string &fl_id);
  CacheStatus GetAllClientEncryptedShares(std::unordered_map<std::string, SharesPb> *value);

  CacheStatus AddClientRestructShares(const std::string &fl_id, const SharesPb &value);
  CacheStatus GetClientRestructShare(const std::string &fl_id, SharesPb *value);
  bool HasClientRestructShare(const std::string &fl_id);
  CacheStatus GetAllClientRestructShares(std::unordered_map<std::string, SharesPb> *value);

  CacheStatus AddClientListSign(const std::string &fl_id, const std::string &value);
  CacheStatus GetClientListSign(const std::string &fl_id, std::string *value);
  bool HasClientListSign(const std::string &fl_id);
  CacheStatus GetAllClientListSign(std::unordered_map<std::string, std::string> *value);

  // fl set value
  CacheStatus AddExchangeKeyClient(const std::string &fl_id);
  bool HasExchangeKeyClient(const std::string &fl_id);
  CacheStatus GetAllExchangeKeyClients(std::vector<std::string> *value);

  CacheStatus AddGetKeysClient(const std::string &fl_id);
  bool HasGetKeysClient(const std::string &fl_id);

  CacheStatus AddShareSecretsClient(const std::string &fl_id);
  bool HasShareSecretsClient(const std::string &fl_id);
  CacheStatus GetAllShareSecretsClients(std::vector<std::string> *value);

  CacheStatus AddGetSecretsClient(const std::string &fl_id);
  bool HasGetSecretsClient(const std::string &fl_id);

  CacheStatus AddUpdateModelClient(const std::string &fl_id);
  bool HasUpdateModelClient(const std::string &fl_id);
  CacheStatus GetAllUpdateModelClients(std::vector<std::string> *value);

  CacheStatus AddGetUpdateModelClient(const std::string &fl_id);
  bool HasGetUpdateModelClient(const std::string &fl_id);

  CacheStatus AddReconstructClient(const std::string &fl_id);
  bool HasReconstructClient(const std::string &fl_id);

  CacheStatus SetClientNoises(const ClientNoises &noises);
  CacheStatus GetClientNoises(ClientNoises *noises);

  bool ResetOnNewIteration();

 private:
  CacheStatus AddPbItem(const std::string &name, const std::string &fl_id, const google::protobuf::Message &value);
  template <class PbType, typename std::enable_if<std::is_base_of_v<google::protobuf::Message, PbType>, int>::type = 0>
  CacheStatus GetPbItem(const std::string &name, const std::string &fl_id, PbType *value) {
    if (value == nullptr) {
      return kCacheInnerErr;
    }
    std::string pb_value;
    auto status = GetPbItem(name, fl_id, &pb_value);
    if (status != kCacheSuccess) {
      return status;
    }
    auto ret = value->ParseFromString(pb_value);
    if (!ret) {
      MS_LOG_ERROR << "Parse string value to protobuf value failed";
      return kCacheInnerErr;
    }
    return kCacheSuccess;
  }
  template <class PbType, typename std::enable_if<std::is_base_of_v<google::protobuf::Message, PbType>, int>::type = 0>
  CacheStatus GetAllPbItems(const std::string &name, std::unordered_map<std::string, PbType> *value) {
    if (value == nullptr) {
      return kCacheInnerErr;
    }
    std::unordered_map<std::string, std::string> pb_items;
    auto status = GetAllPbItems(name, &pb_items);
    if (status != kCacheSuccess) {
      return status;
    }
    std::unordered_map<std::string, PbType> map_ret;
    for (auto &item : pb_items) {
      auto ret = map_ret[item.first].ParseFromString(item.second);
      if (!ret) {
        MS_LOG_ERROR << "Parse string value to protobuf value failed";
        return kCacheInnerErr;
      }
    }
    *value = std::move(map_ret);
    return kCacheSuccess;
  }
  CacheStatus SetPbValue(const std::string &name, const google::protobuf::Message &value);
  template <class PbType, typename std::enable_if<std::is_base_of_v<google::protobuf::Message, PbType>, int>::type = 0>
  CacheStatus GetPbValue(const std::string &name, PbType *value) {
    if (value == nullptr) {
      return kCacheInnerErr;
    }
    std::string pb_value;
    auto status = GetPbValue(name, &pb_value);
    if (status != kCacheSuccess) {
      return status;
    }
    auto ret = value->ParseFromString(pb_value);
    if (!ret) {
      MS_LOG_ERROR << "Parse string value to protobuf value failed";
      return kCacheInnerErr;
    }
    return kCacheSuccess;
  }

  CacheStatus AddPbItem(const std::string &name, const std::string &fl_id, const std::string &value);
  bool HasPbItem(const std::string &name, const std::string &fl_id);
  CacheStatus GetPbItem(const std::string &name, const std::string &fl_id, std::string *value);
  CacheStatus GetAllPbItems(const std::string &name, std::unordered_map<std::string, std::string> *items);
  CacheStatus GetAllClients(const std::string &name, std::vector<std::string> *items);
  CacheStatus AddFlItem(const std::string &name, const std::string &fl_id);
  bool HasFlItem(const std::string &name, const std::string &fl_id);
  CacheStatus SetPbValue(const std::string &name, const std::string &value);
  CacheStatus GetPbValue(const std::string &name, std::string *value);
  std::shared_ptr<RedisClientBase> GetOneClient();
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_CLIENT_INFOS_H_
