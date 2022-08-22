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
#ifndef MINDSPORE_FL_ARCH_CCSRC_VERTICAL_PYTHON_TENSOR_LIST_PY_H_
#define MINDSPORE_FL_ARCH_CCSRC_VERTICAL_PYTHON_TENSOR_LIST_PY_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <memory>
#include <vector>
#include "common/utils/visible.h"
#include "vertical/python/tensor_py.h"

namespace py = pybind11;

namespace mindspore {
namespace fl {
class MS_EXPORT TensorListItemPy {
 public:
  ~TensorListItemPy() = default;
  TensorListItemPy() = default;
  TensorListItemPy(const std::string &name, const std::vector<TensorItemPy> &tensors,
                   const std::vector<TensorListItemPy> &TensorListItems);

  void set_name(const std::string &name);
  std::string name() const;

  void set_tensors(const std::vector<TensorItemPy> &tensors);
  std::vector<TensorItemPy> tensors() const;

  void set_tensor_list_items(const std::vector<TensorListItemPy> &tensorListItems);
  std::vector<TensorListItemPy> tensorListItems() const;

  void add_tensor(const TensorItemPy &tensor);
  void add_tensor_list_item(const TensorListItemPy &tensorListItem);

 private:
  std::string name_;
  std::vector<TensorItemPy> tensors_;
  std::vector<TensorListItemPy> tensorListItems_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_PYTHON_TENSOR_LIST_PY_H_
