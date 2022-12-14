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

#ifndef MINDSPORE_SECRET_SHARING_H
#define MINDSPORE_SECRET_SHARING_H
#include <string>
#include <vector>
#include "openssl/bn.h"
#include "common/utils/log_adapter.h"
#include "common/common.h"

namespace mindspore {
namespace fl {
namespace armour {
struct Share {
  unsigned int index;
  unsigned char *data;
  size_t len;
  ~Share();
};

void secure_zero(uint8_t *s, size_t);
int GetPrime(BIGNUM *prim);

class SecretSharing {
 public:
  explicit SecretSharing(BIGNUM *prim);
  ~SecretSharing();
  // split the input secret into multiple shares
  int Split(int n, const int k, const char *secret, size_t length, const std::vector<Share *> &shares);
  // reconstruct the secret from multiple shares
  int Combine(size_t k, const std::vector<Share *> &shares, uint8_t *secret, size_t *length);
  int CheckShares(Share *share_i, BIGNUM *x_i, BIGNUM *y_i, BIGNUM *denses_i, BIGNUM *nums_i);
  int CheckSum(BIGNUM *sum) const;
  int LagrangeCal(BIGNUM *nums_j, BIGNUM *x_m, BIGNUM *x_j, BIGNUM *denses_j, BIGNUM *tmp, BN_CTX *ctx);
  int InputCheck(size_t k, const std::vector<Share *> &shares, uint8_t *secret, size_t *length) const;
  void ReleaseNum(BIGNUM *bigNum) const;

 private:
  BIGNUM *bn_prim_;
  // addition in finite field
  bool field_add(BIGNUM *z, const BIGNUM *x, const BIGNUM *y, BN_CTX *ctx);
  // multiplication in finite field
  bool field_mult(BIGNUM *z, const BIGNUM *x, const BIGNUM *y, BN_CTX *ctx);
  // subtraction in finite field
  bool field_sub(BIGNUM *z, const BIGNUM *x, const BIGNUM *y, BN_CTX *ctx);
  // convert secret sharing from Share type to BIGNUM type
  bool GetShare(BIGNUM *x, BIGNUM *share, Share *s_share);
  void FreeBNVector(std::vector<BIGNUM *> bns);
};
}  // namespace armour
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_SECRET_SHARING_H
