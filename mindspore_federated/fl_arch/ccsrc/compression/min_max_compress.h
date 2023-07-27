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

#ifndef MINDSPORE_CCSRC_FL_COMPRESSION_MIN_MAX_COMPRESS_H_
#define MINDSPORE_CCSRC_FL_COMPRESSION_MIN_MAX_COMPRESS_H_

#include <string>
#include <vector>
#include "compression/bit_pack.h"
#include "compression/compress_common.h"

namespace mindspore {
namespace fl {
namespace compression {

MS_EXPORT TensorItemPy run_min_max_compress(const std::vector<float>& origin_data, size_t bit_num) {
  TensorItemPy tensor_item_py;

  size_t size = origin_data.size();

  auto temp1 = static_cast<float>(k1 << bit_num) - 1.0f;
  auto temp2 = static_cast<float>(k1 << (bit_num - k1));
  if (temp1 == 0.0f) {
    MS_LOG(EXCEPTION) << "temp1 value is zero, please check!";
  }

  float min_val = FLT_MAX;
  float max_val = -FLT_MAX;
  for (const auto &datum : origin_data) {
    if (datum > max_val) {
      max_val = datum;
    }
    if (datum < min_val) {
      min_val = datum;
    }
  }
  float scale_val = (max_val - min_val) / temp1 + kEps;
  MS_LOG(INFO) << "min_val: " << min_val << " max_val: " << max_val << " scale_val: " << scale_val;
  if (scale_val == 0.0f) {
    MS_LOG(EXCEPTION) << "scale_val is zero.";
  }
  if (bit_num == k8) {
    std::vector<char> compress_data(size);
    for (size_t i = 0; i < size; ++i) {
      auto round_data = std::round((origin_data[i] - min_val) / scale_val - temp2);
      auto int8_data = int8_t(round_data);
      compress_data[i] = int8_data;
    }
    std::string raw_data(compress_data.data(), compress_data.size());
    tensor_item_py.set_raw_data(raw_data);
  } else {
    std::vector<int> fake_compress_data(size);
    for (size_t i = 0; i < size; ++i) {
      auto round_data = std::round((origin_data[i] - min_val) / scale_val - temp2);
      auto int_data = static_cast<int>(round_data);
      fake_compress_data[i] = int_data;
    }
    std::vector<char>packed_data = mindspore::fl::compression::bit_pack(fake_compress_data, bit_num);
    std::string raw_data(packed_data.data(), packed_data.size());
    tensor_item_py.set_raw_data(raw_data);
  }
  tensor_item_py.set_bit_num(bit_num);
  tensor_item_py.set_size(size);
  tensor_item_py.set_min_val(min_val);
  tensor_item_py.set_max_val(max_val);
  tensor_item_py.set_compress_type(kMinMax);
  return tensor_item_py;
}

}  // namespace compression
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMPRESSION_MIN_MAX_COMPRESS_H_
