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
#include "python/feature_py.h"

namespace mindspore {
namespace fl {
FeatureItemPy::FeatureItemPy(const std::string &feature_name, const py::array &data, const std::vector<size_t> &shape,
                             const std::string &dtype, bool requires_aggr)
    : data_(data) {
  weight_.name = feature_name;
  weight_.data = data.data();
  weight_.shape = shape;
  weight_.size = std::accumulate(shape.begin(), shape.end(), sizeof(float), std::multiplies<size_t>());
  weight_.type = dtype;
  weight_.requires_aggr = requires_aggr;
}

static std::vector<ssize_t> GetStrides(const std::vector<ssize_t> &shape, ssize_t item_size) {
  std::vector<ssize_t> strides;
  strides.reserve(shape.size());
  const auto ndim = shape.size();
  for (size_t i = 0; i < ndim; ++i) {
    auto stride = item_size;
    for (size_t j = i + 1; j < ndim; ++j) {
      stride *= shape[j];
    }
    strides.push_back(stride);
  }
  return strides;
}

std::shared_ptr<FeatureItemPy> FeatureItemPy::CreateFeatureFromModel(const ModelItemPtr &model,
                                                                     const WeightItem &weight_item) {
  if (model == nullptr) {
    MS_LOG_EXCEPTION << "Model cannot be nullptr";
  }
  if (model->weight_data.empty()) {
    MS_LOG_EXCEPTION << "Model weight data cannot be empty";
  }
  if (weight_item.type != "fp32") {
    MS_LOG_EXCEPTION << "Data type of model weight can be only be fp32";
  }
  if (weight_item.offset < 0 || weight_item.size <= 0 ||
      weight_item.offset + weight_item.size > model->weight_data.size()) {
    MS_LOG_EXCEPTION << "Weight data offset or size is invalid, offset: " << weight_item.offset
                     << ", size: " << weight_item.size << ", model data size: " << model->weight_data.size();
  }
  const auto &tensor_shape = weight_item.shape;
  std::vector<ssize_t> shape(tensor_shape.begin(), tensor_shape.end());
  auto item_size = sizeof(float);
  auto data = model->weight_data.data() + weight_item.offset;
  auto format = py::format_descriptor<float>::format();

  std::vector<ssize_t> strides = GetStrides(shape, static_cast<ssize_t>(item_size));
  py::buffer_info info(reinterpret_cast<void *>(data), static_cast<ssize_t>(item_size), format,
                       static_cast<ssize_t>(tensor_shape.size()), shape, strides);

  py::object self = py::cast(model);
  py::array buffer_data(py::dtype(info), info.shape, info.strides, info.ptr, self);
  return std::make_shared<FeatureItemPy>(weight_item.name, buffer_data, weight_item.shape, weight_item.type,
                                         weight_item.requires_aggr);
}
}  // namespace fl
}  // namespace mindspore
