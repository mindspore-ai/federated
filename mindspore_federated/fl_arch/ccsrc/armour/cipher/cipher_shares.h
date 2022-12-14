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

#ifndef MINDSPORE_CCSRC_ARMOUR_CIPHER_SHARES_H
#define MINDSPORE_CCSRC_ARMOUR_CIPHER_SHARES_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <utility>
#include <algorithm>
#include "armour/secure_protocol/secret_sharing.h"
#include "common/utils/log_adapter.h"
#include "armour/cipher/cipher_init.h"
#include "armour/cipher/cipher_meta_storage.h"

namespace mindspore {
namespace fl {
namespace armour {

class CipherShares {
 public:
  // initialize: get cipher_init_
  CipherShares() { cipher_init_ = &CipherInit::GetInstance(); }
  ~CipherShares() = default;

  static CipherShares &GetInstance() {
    static CipherShares instance;
    return instance;
  }

  // handle the client's request of share secrets.
  bool ShareSecrets(const int cur_iterator, const schema::RequestShareSecrets *share_secrets_req,
                    const std::shared_ptr<FBBuilder> &share_secrets_resp_builder, const std::string next_req_time);
  // handle the client's request of get secrets.
  bool GetSecrets(const schema::GetShareSecrets *get_secrets_req,
                  const std::shared_ptr<FBBuilder> &get_secrets_resp_builder, const std::string &next_req_time);

  // build response code of share secrets.
  void BuildShareSecretsRsp(const std::shared_ptr<FBBuilder> &share_secrets_resp_builder,
                            const schema::ResponseCode retcode, const std::string &reason,
                            const std::string &next_req_time, const int iteration);
  // build response code of get secrets.
  void BuildGetSecretsRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                          const size_t iteration, const std::string &next_req_time,
                          const std::vector<flatbuffers::Offset<schema::ClientShare>> *encrypted_shares);

 private:
  CipherInit *cipher_init_;  // the parameter of the secure aggregation
};
}  // namespace armour
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_ARMOUR_CIPHER_SHARES_H
