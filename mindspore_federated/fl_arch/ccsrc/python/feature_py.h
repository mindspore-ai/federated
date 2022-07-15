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
#ifndef MINDSPORE_FEDERATED_FEATURE_PY_H
#define MINDSPORE_FEDERATED_FEATURE_PY_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <memory>
#include <vector>
#include "common/common.h"
#include "server/model_store.h"

namespace py = pybind11;

namespace mindspore {
namespace fl {
class FeatureItemPy {
 public:
  FeatureItemPy(const std::string &feature_name, const py::array &data, const std::vector<size_t> &shape,
                const std::string &dtype, bool require_aggr);

  InputWeight GetWeight() const { return weight_; }

  std::string feature_name() const { return weight_.name; }
  py::array data() { return data_; }
  std::vector<size_t> shape() const { return weight_.shape; }
  std::string dtype() const { return weight_.type; }
  bool require_aggr() const { return weight_.require_aggr; }

  static std::shared_ptr<FeatureItemPy> CreateFeatureFromModel(const ModelItemPtr &model,
                                                               const WeightItem &weight_item);

 private:
  py::array data_;
  InputWeight weight_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FEDERATED_FEATURE_PY_H
