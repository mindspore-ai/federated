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
#ifndef MINDSPORE_FL_ARCH_CCSRC_VERTICAL_PYTHON_WORKER_REGISTER_PY_H_
#define MINDSPORE_FL_ARCH_CCSRC_VERTICAL_PYTHON_WORKER_REGISTER_PY_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <memory>
#include "common/utils/visible.h"

namespace py = pybind11;

namespace mindspore {
namespace fl {
class MS_EXPORT WorkerRegisterItemPy {
 public:
  ~WorkerRegisterItemPy() = default;
  WorkerRegisterItemPy() = default;

  void set_worker_name(const std::string &worker_name);
  std::string worker_name() const;

 private:
  std::string worker_name_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_PYTHON_WORKER_REGISTER_PY_H_
