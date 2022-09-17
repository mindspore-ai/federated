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

#ifndef MINDSPORE_FEDERATED_ECC_H
#define MINDSPORE_FEDERATED_ECC_H

#include <vector>
#include <string>

#include "openssl/objects.h"
#include "openssl/rand.h"

#include "armour/base_crypto/base_unit.h"
#include "common/parallel_for.h"

namespace mindspore {
namespace fl {
namespace psi {

class ECC {
 public:
  explicit ECC(const std::string &curve_name) {
    RAND_priv_bytes(private_key_, LENGTH_32);
    nid_ = GetNID(curve_name);
  }

  ECC(const std::string &curve_name, size_t thread_num, size_t chunk_size) : ECC(curve_name) {
    thread_num_ = thread_num;
    chunk_size_ = chunk_size;
  }

  ~ECC() { OPENSSL_cleanse(private_key_, LENGTH_32); }

  static int GetNID(const std::string &curve_name) {
    if (curve_name == "sm2") {
      return NID_sm2;
    } else if (curve_name == "brainpoolP256r1") {
      return NID_brainpoolP256r1;
    } else if (curve_name == "p256") {
      return NID_X9_62_prime256v1;
    } else {
      MS_LOG(ERROR) << "Not support this ECC type: " << curve_name;
      return 0;
    }
  }

  std::vector<std::string> DcpsAndInverseMul(const std::vector<std::string> &compress_vct, size_t compress_length,
                                             size_t compare_length);

  std::vector<std::string> DcpsAndMul(const std::vector<std::string> &compress_vct, size_t compress_length,
                                      size_t compare_length);

  std::vector<std::string> HashToCurveAndMul(const std::vector<std::string> &hash_inputs, size_t compress_length,
                                             size_t compare_length);

  size_t thread_num_ = 1;
  size_t chunk_size_ = 1;

 private:
  int nid_ = NID_sm2;
  uint8_t private_key_[LENGTH_32];
};

}  // namespace psi
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_FEDERATED_ECC_H
