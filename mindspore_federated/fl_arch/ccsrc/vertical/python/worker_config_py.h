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
#ifndef MINDSPORE_FL_ARCH_CCSRC_VERTICAL_PYTHON_WORKER_CONFIG_PY_H_
#define MINDSPORE_FL_ARCH_CCSRC_VERTICAL_PYTHON_WORKER_CONFIG_PY_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <memory>
#include <vector>
#include "common/utils/visible.h"

namespace py = pybind11;

namespace mindspore {
namespace fl {
class MS_EXPORT WorkerConfigItemPy {
 public:
  ~WorkerConfigItemPy() = default;
  WorkerConfigItemPy() = default;

  void set_primary_key(const std::string &primary_key);
  std::string primary_key() const;

  void set_bucket_num(const uint64_t &bucket_num);
  uint64_t bucket_num() const;

  void set_shard_num(const uint64_t &shard_num);
  uint64_t shard_num() const;

  void set_join_type(const std::string &join_type);
  std::string join_type() const;

 private:
  std::string primary_key_;
  uint64_t bucket_num_;
  uint64_t shard_num_;
  std::string join_type_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_PYTHON_WORKER_CONFIG_PY_H_
