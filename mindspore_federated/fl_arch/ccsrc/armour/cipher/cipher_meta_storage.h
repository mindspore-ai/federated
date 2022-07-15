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

#ifndef MINDSPORE_CCSRC_ARMOUR_CIPHER_META_STORAGE_H
#define MINDSPORE_CCSRC_ARMOUR_CIPHER_META_STORAGE_H

#include <utility>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "common/utils/log_adapter.h"
#include "armour/secure_protocol/secret_sharing.h"
#include "schema/fl_job_generated.h"
#include "schema/cipher_generated.h"
#include "common/common.h"

#define IND_IV_INDEX 0
#define PW_IV_INDEX 1
#define PW_SALT_INDEX 2

namespace mindspore {
namespace fl {
namespace armour {

template <typename T1>
bool CreateArray(std::vector<T1> *newData, const flatbuffers::Vector<T1> &fbs_arr) {
  if (newData == nullptr) return false;
  size_t size = newData->size();
  size_t size_fbs_arr = fbs_arr.size();
  if (size != size_fbs_arr) return false;
  for (size_t i = 0; i < size; ++i) {
    newData->at(i) = fbs_arr.Get(i);
  }
  return true;
}

constexpr int SHARE_MAX_SIZE = 256;
constexpr int SECRET_MAX_LEN_DOUBLE = 66;

struct clientshare_str {
  std::string fl_id;
  std::vector<uint8_t> share;
  int index;
};

struct CipherPublicPara {
  int t;
  int g;
  uint8_t prime[PRIME_MAX_LEN];
  uint8_t p[SECRET_MAX_LEN];
  float dp_eps;
  float dp_delta;
  float dp_norm_clip;
  std::string encrypt_type;
  float sign_k;
  float sign_eps;
  float sign_thr_ratio;
  float sign_global_lr;
  int sign_dim_out;
};

class CipherMetaStorage {
 public:
  // Register Prime.
  void RegisterPrime(const std::string &prime);
  // Get tprime from shared server.
  bool GetPrimeFromServer(uint8_t *prime);
  // Get client keys from shared server.
  void GetClientKeysFromServer(std::map<std::string, std::vector<std::vector<uint8_t>>> *clients_keys_list);
  // Get stable secure aggregation's client key from shared server.
  void GetStableClientKeysFromServer(std::map<std::string, std::vector<std::vector<uint8_t>>> *clients_keys_list);
  void GetClientIVsFromServer(std::map<std::string, std::vector<std::vector<uint8_t>>> *clients_ivs_list);
  // Get client noises from shared server.
  bool GetClientNoisesFromServer(std::vector<float> *cur_public_noise);
  // Update client key with signature to shared server.
  bool UpdateClientKeyToServer(const schema::RequestExchangeKeys *exchange_keys_req);
  // Update stable secure aggregation's client key to shared server.
  bool UpdateStableClientKeyToServer(const schema::RequestExchangeKeys *exchange_keys_req);
  // Update client noise to shared server.
  bool UpdateClientNoiseToServer(const std::vector<float> &cur_public_noise);

  // Get client shares from shared server.
  void GetClientReconstructSharesFromServer(std::map<std::string, std::vector<clientshare_str>> *clients_shares_list);
  void GetClientEncryptedSharesFromServer(std::map<std::string, std::vector<clientshare_str>> *clients_shares_list);
  // Update client share to shared server.
  bool UpdateClientReconstructShareToServer(
    const std::string &fl_id, const flatbuffers::Vector<flatbuffers::Offset<schema::ClientShare>> *shares);
  bool UpdateClientEncryptedShareToServer(const std::string &fl_id,
                                          const flatbuffers::Vector<flatbuffers::Offset<schema::ClientShare>> *shares);

 private:
  void GetClientSharesFromServerInner(const std::unordered_map<std::string, fl::SharesPb> &value_map,
                                      std::map<std::string, std::vector<clientshare_str>> *clients_shares_list);
  bool UpdateClientShareToServerInner(const std::string &fl_id,
                                      const flatbuffers::Vector<flatbuffers::Offset<schema::ClientShare>> *shares,
                                      SharesPb *shares_pb);
};
}  // namespace armour
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_ARMOUR_CIPHER_META_STORAGE_H
