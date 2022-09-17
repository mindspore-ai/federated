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

#include <string>

#include "armour/base_crypto/ecc.h"

namespace mindspore {
namespace fl {
namespace psi {
std::vector<std::string> ECC::HashToCurveAndMul(const std::vector<std::string> &hash_inputs, size_t compress_length,
                                                size_t compare_length) {
  time_t time_start;
  time_t time_end;
  time(&time_start);

  std::vector<std::string> p_k_vct(hash_inputs.size());
  ECGroupClass group(nid_);
  BigNumClass bn_priv_Key(std::string(private_key_, private_key_ + LENGTH_32), group.bn_n);
  ParallelSync parallel_sync(thread_num_);
  parallel_sync.parallel_for(0, hash_inputs.size(), chunk_size_, [&](size_t beg, size_t end) {
    for (size_t i = beg; i < end; i++) {
      p_k_vct[i] = ECPointClass::GenPointFromString(group, hash_inputs[i], true)
                     .BNMul(bn_priv_Key)
                     .CompressToString(compress_length)
                     .substr(0, compare_length);
    }
  });

  time(&time_end);
  MS_LOG(INFO) << "Compute p^k, time cost: " << difftime(time_end, time_start) << " s.";
  return p_k_vct;
}

std::vector<std::string> ECC::DcpsAndMul(const std::vector<std::string> &compress_vct, size_t compress_length,
                                         size_t compare_length) {
  time_t time_start;
  time_t time_end;
  time(&time_start);

  size_t input_num = compress_vct.size();
  std::vector<std::string> p_a_b_vector(input_num);
  ECGroupClass group(nid_);
  BigNumClass bn_priv_Key(std::string(private_key_, private_key_ + LENGTH_32), group.bn_n);
  ParallelSync parallel_sync(thread_num_);
  parallel_sync.parallel_for(0, input_num, chunk_size_, [&](size_t beg, size_t end) {
    for (size_t i = beg; i < end; i++) {
      ECPointClass decompress_point(group, compress_vct[i], compress_length);
      p_a_b_vector[i] = decompress_point.BNMul(bn_priv_Key).CompressToString(compress_length).substr(0, compare_length);
    }
  });

  time(&time_end);
  MS_LOG(INFO) << "Bob decompress and compute p1^a^b, time cost: " << difftime(time_end, time_start) << " s.";
  return p_a_b_vector;
}

std::vector<std::string> ECC::DcpsAndInverseMul(const std::vector<std::string> &compress_vct, size_t compress_length,
                                                size_t compare_length) {
  time_t time_start;
  time_t time_end;
  time(&time_start);

  size_t input_num = compress_vct.size();
  std::vector<std::string> p_a_b_bI_vector(input_num);
  ECGroupClass group(nid_);
  BigNumClass bn_priv_Key(std::string(private_key_, private_key_ + LENGTH_32), group.bn_n);
  BigNumClass bn_priv_Key_I = bn_priv_Key.Inverse(group.bn_n);
  ParallelSync parallel_sync(thread_num_);
  parallel_sync.parallel_for(0, input_num, chunk_size_, [&](size_t beg, size_t end) {
    for (size_t i = beg; i < end; i++) {
      ECPointClass decompress_point(group, compress_vct[i], compress_length);
      p_a_b_bI_vector[i] = decompress_point.BNMul(bn_priv_Key_I).CompressToString(LENGTH_32).substr(0, compare_length);
    }
  });

  time(&time_end);
  MS_LOG(INFO) << "Bob decompress and compute p1^a^b^(b^-1), time cost: " << difftime(time_end, time_start) << " s.";
  return p_a_b_bI_vector;
}

}  // namespace psi
}  // namespace fl
}  // namespace mindspore
