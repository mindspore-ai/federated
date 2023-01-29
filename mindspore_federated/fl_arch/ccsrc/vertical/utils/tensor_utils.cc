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

#include "vertical/utils/tensor_utils.h"
#include <vector>
#include <string>

namespace mindspore {
namespace fl {
TensorListItemPy ParseTensorListProto(const TensorListProto &tensorListProto) {
  TensorListItemPy tensorListItemPy;
  std::string name = tensorListProto.name();
  tensorListItemPy.set_name(name);
  auto tensors_proto = tensorListProto.tensors();
  std::vector<TensorItemPy> tensors;
  for (const auto &item : tensors_proto) {
    TensorItemPy tensor;
    tensor.set_name(item.name());
    tensor.set_ref_key(item.ref_key());
    tensor.set_dtype(item.data_type());
    tensor.set_raw_data(item.raw_data());
    tensor.set_compress_type(item.compress_type());
    tensor.set_min_val(item.min_val());
    tensor.set_max_val(item.max_val());
    std::vector<size_t> shape;
    int dims_size = item.dims_size();
    for (int i = 0; i < dims_size; i++) {
      shape.push_back(item.dims(i));
    }
    tensor.set_shape(shape);
    tensors.push_back(tensor);
  }
  tensorListItemPy.set_tensors(tensors);

  int tensor_list_size = tensorListProto.tensor_list_size();
  std::vector<TensorListItemPy> tensorListItems;
  for (int i = 0; i < tensor_list_size; i++) {
    tensorListItems.push_back(ParseTensorListProto(tensorListProto.tensor_list(i)));
  }
  tensorListItemPy.set_tensor_list_items(tensorListItems);
  return tensorListItemPy;
}

void CreateTensorProto(TensorProto *tensor_proto, const TensorItemPy &tensor, std::string ref_key) {
  if (!ref_key.empty()) {
    tensor_proto->set_ref_key(ref_key);
  }
  std::string data_type = tensor.dtype();
  if (data_type.empty()) {
    MS_LOG_EXCEPTION << "CreateTensorProto: input a Tensor with unsupported value type";
  }
  tensor_proto->set_data_type(data_type);

  for (size_t dim : tensor.shape()) {
    tensor_proto->add_dims(dim);
  }
  tensor_proto->set_raw_data(tensor.raw_data());
  tensor_proto->set_compress_type(tensor.compress_type());
  tensor_proto->set_min_val(tensor.min_val());
  tensor_proto->set_max_val(tensor.max_val());
}

void CreateTensorListProto(TensorListProto *tensor_list_proto, const TensorListItemPy &tensorListItemPy) {
  MS_EXCEPTION_IF_NULL(tensor_list_proto);
  tensor_list_proto->set_name(tensorListItemPy.name());

  auto tensors = tensorListItemPy.tensors();
  auto tensorListItems = tensorListItemPy.tensorListItems();

  for (const auto &tensor : tensors) {
    TensorProto *tensor_proto = tensor_list_proto->add_tensors();
    CreateTensorProto(tensor_proto, tensor, tensor.ref_key());
  }

  for (const auto &tensorListItem : tensorListItems) {
    TensorListProto *sub_tensor_list_proto = tensor_list_proto->add_tensor_list();
    CreateTensorListProto(sub_tensor_list_proto, tensorListItem);
  }
}
}  // namespace fl
}  // namespace mindspore
