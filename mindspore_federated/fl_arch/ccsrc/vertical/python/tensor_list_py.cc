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
#include "vertical/python/tensor_list_py.h"
#include <functional>
#include <vector>

namespace mindspore {
namespace fl {
TensorListItemPy::TensorListItemPy(const std::string &name, const std::vector<TensorItemPy> &tensors,
                                   const std::vector<TensorListItemPy> &tensorListItems) {
  name_ = name;
  tensors_ = tensors;
  tensorListItems_ = tensorListItems;
}

std::string TensorListItemPy::name() const { return name_; }

std::vector<TensorItemPy> TensorListItemPy::tensors() const { return tensors_; }

std::vector<TensorListItemPy> TensorListItemPy::tensorListItems() const { return tensorListItems_; }

void TensorListItemPy::set_name(const std::string &name) { name_ = name; }

void TensorListItemPy::set_tensors(const std::vector<TensorItemPy> &tensors) { tensors_ = tensors; }

void TensorListItemPy::set_tensor_list_items(const std::vector<TensorListItemPy> &tensorListItems) {
  tensorListItems_ = tensorListItems;
}

void TensorListItemPy::add_tensor(const TensorItemPy &tensor) { tensors_.push_back(tensor); }

void TensorListItemPy::add_tensor_list_item(const TensorListItemPy &tensorListItem) {
  tensorListItems_.push_back(tensorListItem);
}
}  // namespace fl
}  // namespace mindspore
