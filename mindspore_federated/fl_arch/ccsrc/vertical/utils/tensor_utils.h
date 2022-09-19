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
#ifndef MINDSPORE_FL_ARCH_CCSRC_VERTICAL_UTILS_TENSOR_UTILS_H_
#define MINDSPORE_FL_ARCH_CCSRC_VERTICAL_UTILS_TENSOR_UTILS_H_

#include <vector>
#include <string>
#include "vertical/python/tensor_list_py.h"
#include "vertical/python/tensor_py.h"
#include "common/protos/vfl.pb.h"
#include "common/utils/log_adapter.h"

namespace mindspore {
namespace fl {
TensorListItemPy ParseTensorListProto(const TensorListProto &tensorListProto);

void CreateTensorProto(TensorProto *tensor_proto, const TensorItemPy &tensor, std::string ref_key);

void CreateTensorListProto(TensorListProto *tensor_list_proto, const TensorListItemPy &tensorListItemPy);
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_UTILS_TENSOR_UTILS_H_
