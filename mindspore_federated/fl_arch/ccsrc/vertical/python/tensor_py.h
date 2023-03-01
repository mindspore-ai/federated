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
#ifndef MINDSPORE_FL_ARCH_CCSRC_VERTICAL_PYTHON_TENSOR_PY_H_
#define MINDSPORE_FL_ARCH_CCSRC_VERTICAL_PYTHON_TENSOR_PY_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <memory>
#include <vector>
#include "common/utils/visible.h"

namespace py = pybind11;

namespace mindspore {
namespace fl {
class MS_EXPORT TensorItemPy {
 public:
  ~TensorItemPy() = default;
  TensorItemPy() = default;

  void set_name(const std::string &name);
  std::string name() const;

  void set_ref_key(const std::string &ref_key);
  std::string ref_key() const;

  void set_shape(const std::vector<size_t> &shape);
  std::vector<size_t> shape() const;

  void set_dtype(const std::string &dtype);
  std::string dtype() const;

  void set_raw_data(const std::string &raw_data);
  std::string raw_data() const;

  void set_compress_type(const std::string &compress_type);
  std::string compress_type() const;

  void set_min_val(const float &min_val);
  float min_val() const;

  void set_max_val(const float &max_val);
  float max_val() const;

  void set_size(const size_t &size);
  size_t size() const;

  void set_bit_num(const size_t &bit_num);
  size_t bit_num() const;

  void set_offset(const float &offset);
  float offset() const;

  size_t raw_data_size() const;

 private:
  std::string name_;
  std::string ref_key_;
  std::vector<size_t> shape_;
  std::string dtype_;
  std::string raw_data_;
  std::string compress_type_;
  float min_val_;
  float max_val_;
  size_t size_;
  size_t bit_num_;
  float offset_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_PYTHON_TENSOR_PY_H_
