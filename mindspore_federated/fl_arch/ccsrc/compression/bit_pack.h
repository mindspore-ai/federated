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

#ifndef MINDSPORE_CCSRC_FL_COMPRESSION_BIT_PACK_H_
#define MINDSPORE_CCSRC_FL_COMPRESSION_BIT_PACK_H_

#include <string>
#include <vector>
#include "compression/compress_common.h"

namespace mindspore {
namespace fl {
namespace compression {

std::vector<int> real_vec_to_binary_vec(const std::vector<int>& real_vec, size_t bit_num) {
  std::vector<int> binary_vec;
  binary_vec.resize(real_vec.size() * bit_num);
  size_t index = 0;
  for (const auto &real_data : real_vec) {
    for (size_t j = 0; j < bit_num; ++j) {
      binary_vec[index] = (real_data >> (bit_num - 1 - j)) & 1;
      index++;
    }
  }
  return binary_vec;
}

std::vector<char> binary_vec_to_int8_vec(std::vector<int>* binary_vec) {
  size_t binary_vec_size = binary_vec->size();
  size_t remainder = binary_vec_size % k8;
  if (remainder > 0) {
    auto zero_nums = k8 - remainder;
    for (size_t i = 0; i < zero_nums; ++i) {
      binary_vec->emplace_back(0);
    }
    MS_LOG(INFO) << "Add " << zero_nums << " zeros in fake binary vector is done.";
  }
  std::vector<char> int8_vec(binary_vec->size() / k8);
  size_t index = 0;
  for (char & int8_data : int8_vec) {
    int8_data = static_cast<char>(-k128 * (*binary_vec)[index] +
                                  k64 * (*binary_vec)[index + 1] +
                                  k32 * (*binary_vec)[index + 2] +
                                  k16 * (*binary_vec)[index + 3] +
                                  k8 * (*binary_vec)[index + 4] +
                                  k4 * (*binary_vec)[index + 5] +
                                  k2 * (*binary_vec)[index + 6] +
                                  k1 * (*binary_vec)[index + 7]);
    index += 8;
  }
  return int8_vec;
}

std::vector<char> bit_pack(const std::vector<int>& real_vec, size_t bit_num) {
  std::vector<int> binary_vec = real_vec_to_binary_vec(real_vec, bit_num);
  MS_LOG(INFO) << "Convert real vector to fake binary vector is done, and the size of fake binary vector is: "
               << binary_vec.size();
  std::vector<char> int8_vec = binary_vec_to_int8_vec(&binary_vec);
  return int8_vec;
}

MS_EXPORT TensorItemPy run_bit_pack(const std::vector<float>& real_vec, size_t bit_num) {
  TensorItemPy tensor_item_py;
  size_t real_vec_size = real_vec.size();
  std::vector<int> int_vec(real_vec_size);

  // pre process
  float min_val = FLT_MAX;
  float max_val = -FLT_MAX;
  for (const auto & real_data : real_vec) {
    if (real_data < min_val) {
      min_val = real_data;
    }
    if (real_data > max_val) {
      max_val = real_data;
    }
  }
  auto temp = static_cast<float>(1 << (bit_num - k1));
  float offset = min_val - temp;
  MS_LOG(INFO) << "min_val: " << min_val << " max_val: " << max_val << " offset: " << offset;
  if (max_val + offset > temp - 1.0f) {
    tensor_item_py.set_compress_type(kNoCompress);
    MS_LOG(INFO) << "The input is not suitable for bit packing, due to the range of input () is too big.";
    return tensor_item_py;
  }
  for (size_t i = 0; i < real_vec_size; ++i) {
    auto float_data = real_vec[i] + offset;
    auto int_data = static_cast<int>(float_data);
    auto fake_int_data = static_cast<float>(int_data);
    if (float_data - fake_int_data > kEps || fake_int_data - float_data > kEps) {
      tensor_item_py.set_compress_type(kNoCompress);
      return tensor_item_py;
    }
    int_vec[i] = int_data;
  }
  MS_LOG(INFO) << "Cast float input to int is done.";

  // bit packing
  std::vector<char> int8_vec = bit_pack(int_vec, bit_num);
  MS_LOG(INFO) << "bit_pack complete, the packed vector size is: " << int8_vec.size();

  // set information into tensor_item
  std::string raw_data(int8_vec.data(), int8_vec.size());
  tensor_item_py.set_offset(offset);
  tensor_item_py.set_raw_data(raw_data);
  tensor_item_py.set_compress_type(kBitPack);
  tensor_item_py.set_bit_num(bit_num);
  tensor_item_py.set_size(real_vec_size);
  return tensor_item_py;
}

}  // namespace compression
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMPRESSION_BIT_PACK_H_
