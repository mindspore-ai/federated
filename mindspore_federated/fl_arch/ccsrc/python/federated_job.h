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

#ifndef MINDSPORE_SERVER_MASTER_PY_H
#define MINDSPORE_SERVER_MASTER_PY_H

#include <string>
#include <memory>
#include <map>
#include <vector>
#include "common/common.h"
#include "python/feature_py.h"
#include "common/utils/python_adapter.h"

namespace mindspore {
namespace fl {
class MS_EXPORT FederatedJob {
 public:
  static void StartFederatedServer(const std::vector<std::shared_ptr<FeatureItemPy>> &feature_list,
                                   const uint64_t &recovery_iteration, const py::object &after_stated_callback,
                                   const py::object &before_stopped_callback,
                                   const py::object &on_iteration_end_callback);
  static void StartFederatedScheduler();
  static void InitFederatedWorker();
  static void StopFederatedWorker();

  static bool StartFLJob(size_t data_size);
  static py::dict UpdateAndGetModel(std::map<std::string, std::vector<float>> weight_datas);
  static py::dict PullWeight(const std::vector<std::string> &pull_weight_names);
  static bool PushWeight(const std::map<std::string, std::vector<float>> &weight_datas);
  static bool PushMetrics(float loss, float accuracy);
};
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_SERVER_MASTER_PY_H
