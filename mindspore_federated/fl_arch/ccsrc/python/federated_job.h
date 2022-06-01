/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "common/common.h"
#include "common/utils/python_adapter.h"

namespace mindspore {
namespace fl {

class MS_EXPORT FederatedJob {
 public:
  static void StartFederatedJob();
  static bool StartServerAction();
  static bool StartSchedulerAction();
  static bool StartFLWorkerAction();

  static bool StartFLJob(size_t data_size);
  static py::dict UpdateAndGetModel(std::map<std::string, std::vector<float>> weight_datas);
  static py::dict PullWeight();
  static bool PushWeight(std::map<std::string, std::vector<float>> &weight_datas);
  static bool PushMetrics(float loss, float accuracy);
};
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_SERVER_MASTER_PY_H
