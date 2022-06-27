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

#ifndef MINDSPORE_CCSRC_FL_REDIS_KEYS_H
#define MINDSPORE_CCSRC_FL_REDIS_KEYS_H

#include <string>
#include "distributed_cache/instance_context.h"

namespace mindspore {
namespace fl {
namespace cache {
class RedisKeys {
 public:
  static RedisKeys &GetInstance() {
    static RedisKeys instance;
    return instance;
  }
  inline std::string PrefixIteration() const {
    auto iteration_num = InstanceContext::Instance().iteration_num();
    return Prefix() + std::to_string(iteration_num) + ":";
  }
  inline std::string Prefix() const {
    auto fl_name = InstanceContext::Instance().fl_name();
    auto instance_name = InstanceContext::Instance().instance_name();
    return Prefix(fl_name, instance_name);
  }

  inline std::string Prefix(const std::string &fl_name, const std::string &instance_name) const {
    return "ms_fl:" + fl_name + ":" + instance_name + ":";
  }

  std::string InstanceNameString() const {
    auto fl_name = InstanceContext::Instance().fl_name();
    return InstanceNameString(fl_name);
  }

  std::string ModelInfoString() const {
    auto fl_name = InstanceContext::Instance().fl_name();
    return "ms_fl:" + fl_name + ":ModelInfo:String";
  }

  // for scheduler
  std::string InstanceNameString(const std::string &fl_name) const {
    return "ms_fl:" + fl_name + ":InstanceName:String";
  }
  // count
  std::string CountHash() const { return PrefixIteration() + "count:Hash"; }
  std::string CountPerServerHash(const std::string &count_name) const {
    return PrefixIteration() + "count:" + count_name + ":Hash";
  }
  // timer
  std::string TimerHash() const { return PrefixIteration() + "timer:Hash"; }
  // server
  std::string ServerHash() const {
    auto fl_name = InstanceContext::Instance().fl_name();
    auto instance_name = InstanceContext::Instance().instance_name();
    return ServerHash(fl_name, instance_name);
  }
  std::string ServerHeartbeatString(const std::string &node_id) const {
    auto fl_name = InstanceContext::Instance().fl_name();
    auto instance_name = InstanceContext::Instance().instance_name();
    return ServerHeartbeatString(fl_name, instance_name, node_id);
  }
  std::string ServerRegLockString() const { return Prefix() + "server:regLock:String"; }
  // for scheduler
  std::string ServerHash(const std::string &fl_name, const std::string &instance_name) const {
    return Prefix(fl_name, instance_name) + "server:Hash";
  }
  std::string ServerHeartbeatString(const std::string &fl_name, const std::string &instance_name,
                                    const std::string &node_id) const {
    return Prefix(fl_name, instance_name) + "server:heartbeat:" + node_id + ":String";
  }
  // fl set
  std::string ClientExchangeKeysFlSet() const { return GetFlSetKey("exchangeKeys"); }
  std::string ClientGetKeysFlSet() const { return GetFlSetKey("getKeys"); }
  std::string ClientShareSecretsFlSet() const { return GetFlSetKey("shareSecrets"); }
  std::string ClientUpdateModelFlSet() const { return GetFlSetKey("updateModel"); }
  std::string ClientGetSecretsFlSet() const { return GetFlSetKey("getSecrets"); }
  std::string ClientGetUpdateModelFlSet() const { return GetFlSetKey("getUpdateModel"); }
  std::string ClientReconstructFlSet() const { return GetFlSetKey("reconstruct"); }
  // hash key for pb/fp
  std::string ClientDeviceMetasHash() const { return PrefixIteration() + "client:DeviceMetas:Hash"; }
  std::string ClientKeyAttestationHash() const { return PrefixIteration() + "client:KeyAttestation:Hash"; }
  std::string ClientRestructSharesHash() const { return PrefixIteration() + "client:cipher:RestructShares:Hash"; }
  std::string ClientEncryptedSharesHash() const { return PrefixIteration() + "client:EncryptedShares:Hash"; }
  std::string ClientSignaturesHash() const { return PrefixIteration() + "client:Signatures:Hash"; }
  std::string ClientKeysHash() const { return PrefixIteration() + "client:Keys:Hash"; }

  std::string ClientNoisesString() const { return PrefixIteration() + "client:Noises:String"; }
  std::string ClientPrimeString() const { return PrefixIteration() + "client:Prime:String"; }

  std::string InstanceStatusHash() const {
    auto fl_name = InstanceContext::Instance().fl_name();
    auto instance_name = InstanceContext::Instance().instance_name();
    return InstanceStatusHash(fl_name, instance_name);
  }
  std::string HyperParamsString() const {
    auto fl_name = InstanceContext::Instance().fl_name();
    auto instance_name = InstanceContext::Instance().instance_name();
    return HyperParamsString(fl_name, instance_name);
  }

  // for scheduler
  std::string InstanceStatusHash(const std::string &fl_name, const std::string &instance_name) const {
    return Prefix(fl_name, instance_name) + "status:Hash";
  }
  std::string HyperParamsString(const std::string &fl_name, const std::string &instance_name) const {
    return Prefix(fl_name, instance_name) + "hyperParams:String";
  }

  std::string IterationSummaryHash() const { return PrefixIteration() + ":summary:Hash"; }
  std::string IterationSummaryLockString() const { return PrefixIteration() + ":summaryLock:String"; }

  // worker
  std::string WorkerHash(const std::string &fl_name, const std::string &instance_name) const {
    return Prefix(fl_name, instance_name) + "worker:Hash";
  }
  std::string WorkerHeartbeatString(const std::string &fl_name, const std::string &instance_name,
                                    const std::string &node_id) const {
    return Prefix(fl_name, instance_name) + "worker:heartbeat:" + node_id + ":String";
  }

 private:
  std::string model_name_;
  std::string GetFlSetKey(const std::string &set_name) const {
    return PrefixIteration() + "client:cipher:" + set_name + ":flSet";
  }
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_REDIS_KEYS_H
