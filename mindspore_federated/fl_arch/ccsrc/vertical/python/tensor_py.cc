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
#include "vertical/python/tensor_py.h"
#include <functional>
#include <vector>

namespace mindspore {
namespace fl {
void TensorItemPy::set_name(const std::string &name) { name_ = name; }
std::string TensorItemPy::name() const { return name_; }

void TensorItemPy::set_ref_key(const std::string &ref_key) { ref_key_ = ref_key; }
std::string TensorItemPy::ref_key() const { return ref_key_; }

void TensorItemPy::set_shape(const std::vector<size_t> &shape) { shape_ = shape; }
std::vector<size_t> TensorItemPy::shape() const { return shape_; }

void TensorItemPy::set_dtype(const std::string &dtype) { dtype_ = dtype; }
std::string TensorItemPy::dtype() const { return dtype_; }

void TensorItemPy::set_raw_data(const std::string &raw_data) { raw_data_ = raw_data; }
std::string TensorItemPy::raw_data() const { return raw_data_; }

void TensorItemPy::set_compress_type(const std::string &compress_type) { compress_type_ = compress_type; }
std::string TensorItemPy::compress_type() const { return compress_type_; }

void TensorItemPy::set_min_val(const float &min_val) { min_val_ = min_val; }
float TensorItemPy::min_val() const { return min_val_; }

void TensorItemPy::set_max_val(const float &max_val) { max_val_ = max_val; }
float TensorItemPy::max_val() const { return max_val_; }

void TensorItemPy::set_size(const size_t &size) { size_ = size; }
size_t TensorItemPy::size() const { return size_; }

void TensorItemPy::set_bit_num(const size_t &bit_num) { bit_num_ = bit_num; }
size_t TensorItemPy::bit_num() const { return bit_num_; }

void TensorItemPy::set_offset(const float &offset) { offset_ = offset; }
float TensorItemPy::offset() const { return offset_; }

size_t TensorItemPy::raw_data_size() const { return raw_data_.size(); }

}  // namespace fl
}  // namespace mindspore
