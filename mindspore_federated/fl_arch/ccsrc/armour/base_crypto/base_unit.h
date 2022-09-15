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

#ifndef MINDSPORE_FEDERATED_BASE_UNIT_H
#define MINDSPORE_FEDERATED_BASE_UNIT_H

#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/sha.h>

#include <memory>
#include <string>

#include "common/utils/log_adapter.h"

namespace mindspore {
namespace fl {
namespace psi {

constexpr size_t LENGTH_12 = 12;
constexpr size_t LENGTH_32 = 32;
constexpr size_t LENGTH_33 = 33;

struct BNCtxFree {
 public:
  void operator()(BN_CTX *bn_ctx) { BN_CTX_free(bn_ctx); }
};

struct BNFree {
 public:
  void operator()(BIGNUM *bn) { BN_clear_free(bn); }
};

struct ECGroupFree {
 public:
  void operator()(EC_GROUP *group) { EC_GROUP_free(group); }
};

struct ECPointFree {
 public:
  void operator()(EC_POINT *point) { EC_POINT_clear_free(point); }
};

using BnCtxPtr = std::unique_ptr<BN_CTX, BNCtxFree>;
using BigNumPtr = std::unique_ptr<BIGNUM, BNFree>;
using ECGroupPtr = std::unique_ptr<EC_GROUP, ECGroupFree>;
using ECPointPtr = std::unique_ptr<EC_POINT, ECPointFree>;

struct BigNumClass {
  BigNumClass() : bn_ptr(BN_new()) {}

  explicit BigNumClass(const std::string &id_string) : bn_ptr(BN_new()) { FromString(id_string); }

  BigNumClass(const std::string &id_string, const BigNumClass &p) : bn_ptr(BN_new()) { FromString(id_string, p); }

  BigNumClass Inverse(const BigNumClass &p) const {
    BnCtxPtr bn_ctx(BN_CTX_new());
    BigNumClass bn_inv;
    BN_mod_inverse(bn_inv.get(), bn_ptr.get(), p.get(), bn_ctx.get());
    return bn_inv;
  }

  BIGNUM *get() { return bn_ptr.get(); }

  const BIGNUM *get() const { return bn_ptr.get(); }

  std::string ToString() const {
    std::string bn_string(LENGTH_32, '\0');
    BN_bn2binpad(bn_ptr.get(), reinterpret_cast<uint8_t *>(bn_string.data()), LENGTH_32);
    return bn_string;
  }

  void FromString(const std::string &key_string) const {
    if (key_string.size() != LENGTH_32) {
      MS_LOG(ERROR) << "ERROR, input string length is " << key_string.size() << ", not equal to " << LENGTH_32;
    } else {
      BN_bin2bn((const uint8_t *)key_string.data(), LENGTH_32, bn_ptr.get());
    }
  }

  void FromString(const std::string &id_string, const BigNumClass &p) const {
    BigNumClass bn_id(id_string);
    BnCtxPtr bn_ctx(BN_CTX_new());
    BN_nnmod(bn_ptr.get(), bn_id.get(), p.get(), bn_ctx.get());
  }

  BigNumPtr bn_ptr;
};

struct ECGroupClass {
  explicit ECGroupClass(EC_GROUP *group) : group_ptr(group) {
    BnCtxPtr bn_ctx(BN_CTX_new());
    EC_GROUP_get_curve(group_ptr.get(), bn_p.get(), bn_a.get(), bn_b.get(), bn_ctx.get());
    EC_GROUP_get_order(group_ptr.get(), bn_n.get(), bn_ctx.get());
  }

  explicit ECGroupClass(int nid = NID_sm2) : ECGroupClass(EC_GROUP_new_by_curve_name(nid)) {}

  const EC_GROUP *get() const { return group_ptr.get(); }

  BigNumClass bn_a;
  BigNumClass bn_b;
  BigNumClass bn_p;
  BigNumClass bn_n;
  ECGroupPtr group_ptr;
};

struct ECPointClass {
  explicit ECPointClass(const ECGroupClass &ec_group)
      : this_group(ec_group), point_ptr(EC_POINT_new(this_group.get())) {}

  ECPointClass(const ECGroupClass &ec_group, const BigNumClass &bn_key)
      : this_group(ec_group), point_ptr(EC_POINT_new(this_group.get())) {
    BnCtxPtr bn_ctx(BN_CTX_new());
    EC_POINT_mul(this_group.get(), point_ptr.get(), bn_key.get(), NULL, NULL, bn_ctx.get());
  }

  ECPointClass(const ECGroupClass &ec_group, const std::string &compress_p_x, size_t compress_length)
      : this_group(ec_group), point_ptr(EC_POINT_new(this_group.get())) {
    DecompressToPoint(compress_p_x, compress_length);
  }

  // HashToCurve
  // if add_or_rehash is true, we'll use +1 method to solve the problem that is not on the curve.
  static ECPointClass GenPointFromString(const ECGroupClass &group, const std::string &id_string, bool add_or_rehash) {
    BnCtxPtr bn_ctx(BN_CTX_new());
    ECPointClass point(group);
    BigNumClass bn_x(id_string, group.bn_p);
    size_t try_times = 0;
    constexpr size_t MAX_TRY_TIMES = 1000;
    while (true) {
      // try to get y by using x, then do (x,y)->point
      int has_y = EC_POINT_set_compressed_coordinates(group.get(), point.get(), bn_x.get(), 0, bn_ctx.get());
      if (has_y == 1) break;
      if (try_times++ >= MAX_TRY_TIMES) {
        MS_LOG(ERROR) << "Try times >= MAX_TRY_TIMES, Hash_To_Curve Failed.";
        break;
      }
      if (add_or_rehash) {
        BN_add_word(bn_x.get(), 1);
        BN_mod(bn_x.get(), bn_x.get(), group.bn_p.get(), bn_ctx.get());
      } else {
        // reHash
        std::string bn_x_string = bn_x.ToString();
        std::string hash_result(LENGTH_32, '\0');
        SHA256((const uint8_t *)bn_x_string.data(), bn_x_string.size(), (uint8_t *)hash_result.data());
        bn_x.FromString(hash_result, group.bn_p);
      }
    }
    return point;
  }

  std::string CompressToString(size_t compress_length) {
    if (compress_length == LENGTH_33) {
      return CompressToStringByOpenssl();
    } else if (compress_length == LENGTH_32) {
      return CompressToStringByX();
    } else {
      MS_LOG(ERROR) << "Compress length option is ERROR!, input value is " << compress_length;
      return NULL;
    }
  }

  std::string CompressToStringByOpenssl() {
    BnCtxPtr bn_ctx(BN_CTX_new());
    std::string compress_p_a(LENGTH_33, '\0');
    EC_POINT_point2oct(this_group.get(), point_ptr.get(), POINT_CONVERSION_COMPRESSED,
                       reinterpret_cast<uint8_t *>(compress_p_a.data()), LENGTH_33, bn_ctx.get());
    return compress_p_a;
  }

  std::string CompressToStringByX() {
    BnCtxPtr bn_ctx(BN_CTX_new());
    BigNumClass bn_x;
    BigNumClass bn_y;
    EC_POINT_get_affine_coordinates(this_group.get(), point_ptr.get(), bn_x.get(), bn_y.get(), bn_ctx.get());
    auto compress_p_a = bn_x.ToString();
    return compress_p_a;
  }

  void DecompressToPoint(const std::string &compress_p_a, size_t compress_length) {
    if (compress_length == LENGTH_33) {
      DecompressToPointByOpenssl(compress_p_a);
    } else if (compress_length == LENGTH_32) {
      DecompressToPointByX(compress_p_a);
    } else {
      MS_LOG(ERROR) << "Compress length option is ERROR!, input value is " << compress_length;
    }
  }

  void DecompressToPointByOpenssl(const std::string &compress_p_a) {
    if (compress_p_a.length() != LENGTH_33) {
      MS_LOG(ERROR) << "Decompress length option is ERROR!, input value is " << compress_p_a.length()
                    << ", not equal to " << LENGTH_33;
      return;
    }
    BnCtxPtr bn_ctx(BN_CTX_new());
    EC_POINT_oct2point(this_group.get(), point_ptr.get(), reinterpret_cast<const uint8_t *>(compress_p_a.data()),
                       LENGTH_33, bn_ctx.get());
  }

  void DecompressToPointByX(const std::string &compress_p_a) {
    if (compress_p_a.length() != LENGTH_32) {
      MS_LOG(ERROR) << "Input length is " << compress_p_a.length() << ", not equal to" << LENGTH_32;
      return;
    }
    BnCtxPtr bn_ctx(BN_CTX_new());
    BigNumClass bn_x(compress_p_a);
    EC_POINT_set_compressed_coordinates(this_group.get(), point_ptr.get(), bn_x.get(), 0, bn_ctx.get());
  }

  ECPointClass BNMul(const BigNumClass &bn_key) {
    BnCtxPtr bn_ctx(BN_CTX_new());
    ECPointClass r_point(this_group);
    EC_POINT_mul(this_group.get(), r_point.get(), nullptr, point_ptr.get(), bn_key.get(), bn_ctx.get());
    return r_point;
  }

  EC_POINT *get() const { return point_ptr.get(); }

  const ECGroupClass &this_group;

  ECPointPtr point_ptr;
};

}  // namespace psi
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_FEDERATED_BASE_UNIT_H
