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

#ifndef MINDSPORE_FL_ARCH_CCSRC_VERTICAL_VERTICAL_SERVER_COMMUNICATOR_H_
#define MINDSPORE_FL_ARCH_CCSRC_VERTICAL_VERTICAL_SERVER_COMMUNICATOR_H_

#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "vertical/communicator/abstract_communicator.h"
#include "vertical/common.h"
#include "common/protos/data_join.pb.h"
#include "vertical/communicator/trainer_communicator.h"
#include "vertical/communicator/data_join_communicator.h"
#include "vertical/communicator/psi_communicator.h"

namespace mindspore {
namespace fl {
class VerticalServer {
 public:
  VerticalServer() = default;
  ~VerticalServer() = default;

  static VerticalServer &GetInstance();

  void InitVerticalConfigs();

  bool StartVerticalCommunicator();

  void InitVerticalCommunicator(const std::shared_ptr<HttpCommunicator> &http_communicator);

  std::map<std::string, std::shared_ptr<AbstractCommunicator>> &communicators();

  bool Send(const std::string &target_server_name, const TensorListItemPy &tensorListItemPy);

  bool Send(const std::string &target_server_name, const psi::BobPb &bobPb);

  bool Send(const std::string &target_server_name, const psi::ClientPSIInit &clientPSIInit);

  bool Send(const std::string &target_server_name, const psi::ServerPSIInit &serverPSIInit);

  bool Send(const std::string &target_server_name, const psi::BobAlignResult &bobAlignResult);

  bool Send(const std::string &target_server_name, const psi::AlicePbaAndBF &alicePbaAndBF);

  bool Send(const std::string &target_server_name, const psi::AliceCheck &aliceCheck);

  bool Send(const std::string &target_server_name, const psi::PlainData &plainData);

  WorkerConfigItemPy Send(const std::string &target_server_name, const WorkerRegisterItemPy &workerRegisterItem);

  void Receive(const std::string &target_server_name, TensorListItemPy *tensorListItemPy);

  void Receive(const std::string &target_server_name, psi::BobPb *bobPb);

  void Receive(const std::string &target_server_name, psi::ClientPSIInit *clientPSIInit);

  void Receive(const std::string &target_server_name, psi::ServerPSIInit *serverPSIInit);

  void Receive(const std::string &target_server_name, psi::BobAlignResult *bobAlignResult);

  void Receive(const std::string &target_server_name, psi::AlicePbaAndBF *alicePbaAndBF);

  void Receive(const std::string &target_server_name, psi::AliceCheck *aliceCheck);

  void Receive(const std::string &target_server_name, psi::PlainData *plainData);

  bool DataJoinWaitForStart();

 private:
  std::atomic_bool running_ = false;

  std::vector<VerticalConfig> vertical_config_ = {};

  std::map<std::string, std::shared_ptr<AbstractCommunicator>> communicators_;

  std::shared_ptr<HttpCommunicator> http_communicator_ = nullptr;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_VERTICAL_SERVER_COMMUNICATOR_H_
