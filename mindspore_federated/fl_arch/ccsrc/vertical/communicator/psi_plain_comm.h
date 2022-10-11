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

#ifndef MINDSPORE_FEDERATED_PSI_PLAIN_COMM_H
#define MINDSPORE_FEDERATED_PSI_PLAIN_COMM_H

#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "vertical/communicator/abstract_communicator.h"
#include "vertical/common.h"
#include "common/protos/data_join.pb.h"
#include "vertical/communicator/message_queue.h"
#include "vertical/utils/psi_utils.h"

namespace mindspore {
namespace fl {
class PlainDataCommunicator : public AbstractCommunicator {
 public:
  PlainDataCommunicator() = default;
  ~PlainDataCommunicator() = default;

  bool LaunchMsgHandler(const std::shared_ptr<MessageHandler> &message) override;

  void InitCommunicator(const std::shared_ptr<HttpCommunicator> &http_communicator) override;

  bool Send(const std::string &target_server_name, const psi::PlainData &plain_data);

  psi::PlainData Receive(const std::string &target_server_name);

 private:
  bool VerifyProtoMessage(const psi::PlainData &plain_data);

  std::mutex message_received_mutex_;

  std::map<std::string, std::shared_ptr<MessageQueue<psi::PlainData>>> message_queues_ = {};
};
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_FEDERATED_PSI_PLAIN_COMM_H
