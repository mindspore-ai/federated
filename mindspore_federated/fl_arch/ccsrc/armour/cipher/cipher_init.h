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

#ifndef MINDSPORE_CCSRC_ARMOUR_CIPHER_INIT_H
#define MINDSPORE_CCSRC_ARMOUR_CIPHER_INIT_H

#include <vector>
#include <string>
#include "armour/secure_protocol/secret_sharing.h"
#include "common/utils/log_adapter.h"
#include "schema/fl_job_generated.h"
#include "schema/cipher_generated.h"
#include "armour/cipher/cipher_meta_storage.h"

namespace mindspore {
namespace fl {
namespace armour {
struct CipherConfig {
  float share_secrets_ratio = 1.0;
  uint64_t cipher_time_window = 300000;
  uint64_t exchange_keys_threshold = 0;
  uint64_t get_keys_threshold = 0;
  uint64_t share_secrets_threshold = 0;
  uint64_t get_secrets_threshold = 0;
  uint64_t get_client_list_threshold = 0;
  uint64_t push_list_sign_threshold = 0;
  uint64_t get_list_sign_threshold = 0;
  uint64_t minimum_clients_for_reconstruct = 0;
  uint64_t minimum_secret_shares_for_reconstruct = 0;
};

// Initialization of secure aggregation.
class CipherInit {
 public:
  static CipherInit &GetInstance() {
    static CipherInit instance;
    return instance;
  }

  // Initialize the parameters of the secure aggregation.
  bool Init(const CipherPublicPara &param, size_t time_out_mutex, const CipherConfig &cipher_config);

  bool ReInitForScaling();

  // Get public params. which is given to start fl job thread.
  CipherPublicPara *GetPublicParams() { return &publicparam_; }

  size_t share_secrets_threshold;            // the minimum number of clients to share
                                             // secret fragments.
  size_t clients_threshold_for_reconstruct;  // the minimum number of clients to
                                             // reconstruct secret mask.
  size_t exchange_key_threshold;             // the minimum number of clients to send public
                                             // keys.
  size_t push_list_sign_threshold;           // the minimum number of clients to push
                                             // client list signature.
  size_t secrets_minnums_;                   // the minimum number of secret fragment s to
                                             // reconstruct secret mask.
  size_t featuremap_;                        // the size of data to deal.
  CipherPublicPara publicparam_;             // the param containing encrypted public parameters.
  CipherMetaStorage cipher_meta_storage_;

 private:
  size_t client_list_threshold;    // the minimum number of clients to get update
                                   // model client list.
  size_t get_key_threshold;        // the minimum number of clients to get public keys.
  size_t get_list_sign_threshold;  // the minimum number of clients to get client
                                   // list signature.
  size_t get_secrets_threshold;    // the minimum number of clients to get secret
                                   // fragments.
  size_t time_out_mutex_;          // timeout mutex.

  // Check whether the parameters are valid.
  bool Check_Parames();

  std::string new_prime_;
};
}  // namespace armour
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_ARMOUR_CIPHER_COMMON_H
