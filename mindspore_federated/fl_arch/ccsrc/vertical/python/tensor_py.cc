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

void TensorItemPy::set_data(const std::vector<float> &data) { data_ = data; }
std::vector<float> TensorItemPy::data() const { return data_; }
}  // namespace fl
}  // namespace mindspore
