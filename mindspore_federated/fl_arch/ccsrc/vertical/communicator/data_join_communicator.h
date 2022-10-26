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

#ifndef MINDSPORE_FL_ARCH_CCSRC_VERTICAL_DATA_JOIN_COMMUNICATOR_H_
#define MINDSPORE_FL_ARCH_CCSRC_VERTICAL_DATA_JOIN_COMMUNICATOR_H_

#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "vertical/communicator/abstract_communicator.h"
#include "vertical/common.h"
#include "vertical/utils/data_join_utils.h"
#include "vertical/python/worker_config_py.h"
#include "vertical/python/worker_register_py.h"

namespace mindspore {
namespace fl {
class DataJoinCommunicator : public AbstractCommunicator {
 public:
  DataJoinCommunicator() = default;
  ~DataJoinCommunicator() = default;

  WorkerConfigItemPy Send(const std::string &target_server_name, const WorkerRegisterItemPy &workerRegisterItemPy);

  bool LaunchMsgHandler(const std::shared_ptr<MessageHandler> &message) override;

  void InitCommunicator(const std::shared_ptr<HttpCommunicator> &http_communicator) override;

  bool waitForRegister(const uint32_t &timeout);

  void notifyForRegister();

  void SendWorkerConfig(const std::shared_ptr<MessageHandler> &message);

 private:
  bool VerifyProtoMessage(const WorkerRegisterItemPy &workerRegisterItem);

  std::condition_variable message_received_cond_;

  std::mutex message_received_mutex_;

  bool is_worker_registered_ = false;

  WorkerConfigItemPy workerConfigItemPy_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_DATA_JOIN_COMMUNICATOR_H_
