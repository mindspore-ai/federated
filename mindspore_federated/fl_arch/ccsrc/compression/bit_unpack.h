/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_COMPRESSION_BIT_UNPACK_H_
#define MINDSPORE_CCSRC_FL_COMPRESSION_BIT_UNPACK_H_

#include <vector>
#include "compression/compress_common.h"

namespace mindspore {
namespace fl {
namespace compression {

template <typename T>
std::vector<int> int8_vec_to_binary_vec(const T& int8_vec) {
  std::vector<int> binary_vec;
  binary_vec.resize(int8_vec.size() * k8);
  size_t index = 0;
  for (const auto &int8_data : int8_vec) {
    for (size_t j = 0; j < k8; ++j) {
      binary_vec[index] = (int8_data >> (k7 - j)) & 1;
      index++;
    }
  }
  return binary_vec;
}

std::vector<int> binary_vec_to_real_vec(const std::vector<int>& binary_vec, size_t bit_num) {
  std::vector<int> real_vec(binary_vec.size() / bit_num);
  size_t index = 0;
  for (int & real_data : real_vec) {
    real_data = -(1 << (bit_num - 1)) * binary_vec[index];
    index++;
    for (size_t j = 1; j < bit_num; ++j) {
      real_data += (1 << (bit_num - 1 - j)) * binary_vec[index];
      index++;
    }
  }
  return real_vec;
}

template <typename T>
std::vector<int> bit_unpack(const T& int8_vec, size_t bit_num) {
  std::vector<int> binary_vec = int8_vec_to_binary_vec(int8_vec);
  MS_LOG(INFO) << "Convert packed vector to fake binary vector is done, and the size of fake binary vector is: "
               << binary_vec.size();
  std::vector<int> real_vec = binary_vec_to_real_vec(binary_vec, bit_num);
  return real_vec;
}

MS_EXPORT std::vector<float> run_bit_unpack(const TensorItemPy& tensor_item_py) {
  // get information from tensor_item
  size_t bit_num = tensor_item_py.bit_num();
  float offset = tensor_item_py.offset();
  auto raw_data = tensor_item_py.raw_data();

  // bit packing
  std::vector<int> int_vec = bit_unpack(raw_data, bit_num);
  MS_LOG(INFO) << "bit_unpack complete, the unpacked vector size is: " << int_vec.size();

  // post process
  size_t size = tensor_item_py.size();
  if (int_vec.size() < size) {
    MS_LOG(ERROR) << "input is not enough to be unpacked.";
    return {};
  }
  std::vector<float> real_vec(size);
  for (size_t i = 0; i < size; ++i) {
    real_vec[i] = static_cast<float>(int_vec[i]) - offset;
  }
  return real_vec;
}

}  // namespace compression
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMPRESSION_BIT_UNPACK_H_
