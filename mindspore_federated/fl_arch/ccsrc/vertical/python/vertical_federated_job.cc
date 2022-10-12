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

#include "vertical/python/vertical_federated_job.h"

#include <vector>
#include <string>
#include "vertical/communicator/trainer_communicator.h"
#include "vertical/vertical_server.h"
#include "vertical/common.h"

namespace mindspore {
namespace fl {
void VerticalFederatedJob::StartVerticalCommunicator() { VerticalServer::GetInstance().StartVerticalCommunicator(); }

void VerticalFederatedJob::Send(const std::string &target_server_name, const TensorListItemPy &tensorListItemPy) {
  VerticalServer::GetInstance().Send(target_server_name, tensorListItemPy);
}

TensorListItemPy VerticalFederatedJob::Receive(const std::string &target_server_name) {
  TensorListItemPy tensorListItemPy;
  VerticalServer::GetInstance().Receive(target_server_name, &tensorListItemPy);
  return tensorListItemPy;
}
}  // namespace fl
}  // namespace mindspore
