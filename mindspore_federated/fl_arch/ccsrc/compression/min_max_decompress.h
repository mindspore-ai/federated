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

#ifndef MINDSPORE_CCSRC_FL_COMPRESSION_MIN_MAX_DECOMPRESS_H_
#define MINDSPORE_CCSRC_FL_COMPRESSION_MIN_MAX_DECOMPRESS_H_

#include <string>
#include <vector>
#include "compression/bit_unpack.h"
#include "compression/compress_common.h"

namespace mindspore {
namespace fl {
namespace compression {

MS_EXPORT std::vector<float> run_min_max_decompress(const TensorItemPy& tensor_item_py) {
  size_t bit_num = tensor_item_py.bit_num();
  float min_val = tensor_item_py.min_val();
  float max_val = tensor_item_py.max_val();

  auto temp1 = static_cast<float>(k1 << bit_num) - 1.0f;
  auto temp2 = static_cast<float>(k1 << (bit_num - k1));
  float scale_val = static_cast<float>(max_val - min_val) / temp1 + kEps;
  MS_LOG(INFO) << "min_val: " << min_val << " max_val: " << max_val << " scale_val: " << scale_val;

  std::string raw_data = tensor_item_py.raw_data();
  size_t size = tensor_item_py.size();
  std::vector<float> decompress_data(size);
  if (bit_num == k8) {
    for (size_t i = 0; i < size; ++i) {
      decompress_data[i] = (static_cast<float>(raw_data[i]) + temp2) * scale_val + min_val;
    }
  } else {
    std::vector<int> real_vec = mindspore::fl::compression::bit_unpack(raw_data, bit_num);
    if (real_vec.size() < size) {
      MS_LOG(ERROR) << "The vector from remote cannot be decompressed.";
      return decompress_data;
    }
    for (size_t i = 0; i < size; ++i) {
      decompress_data[i] = (static_cast<float>(real_vec[i]) + temp2) * scale_val + min_val;
    }
  }
  return decompress_data;
}

}  // namespace compression
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMPRESSION_MIN_MAX_DECOMPRESS_H_
