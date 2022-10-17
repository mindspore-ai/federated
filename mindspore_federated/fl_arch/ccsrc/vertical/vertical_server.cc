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

#include "vertical/vertical_server.h"
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace mindspore {
namespace fl {
VerticalServer &VerticalServer::GetInstance() {
  static VerticalServer instance;
  return instance;
}

void VerticalServer::InitVerticalConfigs() {
  std::vector<VerticalConfig> vertical_config = {{KTrainer},       {KBobPb},         {KClientPSIInit},
                                                 {KServerPSIInit}, {KAlicePbaAndBF}, {KBobAlignResult},
                                                 {KAliceCheck},    {KPlainData},     {KDataJoin}};
  vertical_config_ = vertical_config;
}

bool VerticalServer::StartVerticalCommunicator() {
  try {
    if (running_.load()) {
      return true;
    }
    running_ = true;
    InitVerticalConfigs();
    const auto &http_comm = AbstractCommunicator::CreateHttpCommunicator();
    InitVerticalCommunicator(http_comm);
    AbstractCommunicator::StartHttpServer(http_comm);
    MS_LOG(INFO) << "Psi communicator started successfully.";
  } catch (const std::exception &e) {
    MS_LOG_WARNING << "Catch exception and begin exit, exception: " << e.what();
    return false;
  }
  return true;
}

void VerticalServer::InitVerticalCommunicator(const std::shared_ptr<HttpCommunicator> &http_communicator) {
  if (http_communicator == nullptr) {
    MS_LOG(EXCEPTION) << "Communicators for vertical communicator is nullptr.";
  }
  for (const auto &config : vertical_config_) {
    std::string name = config.name;
    std::shared_ptr<AbstractCommunicator> vertical_comm = nullptr;
    if (name == KTrainer) {
      vertical_comm = std::make_shared<TrainerCommunicator>();
    } else if (name == KBobPb) {
      vertical_comm = std::make_shared<BobPbCommunicator>();
    } else if (name == KClientPSIInit) {
      vertical_comm = std::make_shared<ClientPSIInitCommunicator>();
    } else if (name == KServerPSIInit) {
      vertical_comm = std::make_shared<ServerPSIInitCommunicator>();
    } else if (name == KAlicePbaAndBF) {
      vertical_comm = std::make_shared<AlicePbaAndBFCommunicator>();
    } else if (name == KBobAlignResult) {
      vertical_comm = std::make_shared<BobAlignResultCommunicator>();
    } else if (name == KAliceCheck) {
      vertical_comm = std::make_shared<AliceCheckCommunicator>();
    } else if (name == KPlainData) {
      vertical_comm = std::make_shared<PlainDataCommunicator>();
    } else if (name == KDataJoin) {
      vertical_comm = std::make_shared<DataJoinCommunicator>();
    } else {
      MS_LOG(EXCEPTION) << "Vertical config is not valid.";
    }
    vertical_comm->InitCommunicator(http_communicator);
    communicators_[config.name] = vertical_comm;
  }
  http_communicator_ = http_communicator;
}

std::map<std::string, std::shared_ptr<AbstractCommunicator>> &VerticalServer::communicators() { return communicators_; }

bool VerticalServer::Send(const std::string &target_server_name, const TensorListItemPy &tensorListItemPy) {
  auto communicator_ptr = reinterpret_cast<TrainerCommunicator *>(communicators_[KTrainer].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  return communicator_ptr->Send(target_server_name, tensorListItemPy);
}

bool VerticalServer::Send(const std::string &target_server_name, const psi::BobPb &bobPb) {
  auto communicator_ptr = reinterpret_cast<BobPbCommunicator *>(communicators_[KBobPb].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  return communicator_ptr->Send(target_server_name, bobPb);
}

bool VerticalServer::Send(const std::string &target_server_name, const psi::ClientPSIInit &clientPSIInit) {
  auto communicator_ptr = reinterpret_cast<ClientPSIInitCommunicator *>(communicators_[KClientPSIInit].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  return communicator_ptr->Send(target_server_name, clientPSIInit);
}

bool VerticalServer::Send(const std::string &target_server_name, const psi::ServerPSIInit &serverPSIInit) {
  auto communicator_ptr = reinterpret_cast<ServerPSIInitCommunicator *>(communicators_[KServerPSIInit].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  return communicator_ptr->Send(target_server_name, serverPSIInit);
}

bool VerticalServer::Send(const std::string &target_server_name, const psi::BobAlignResult &bobAlignResult) {
  auto communicator_ptr = reinterpret_cast<BobAlignResultCommunicator *>(communicators_[KBobAlignResult].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  return communicator_ptr->Send(target_server_name, bobAlignResult);
}

bool VerticalServer::Send(const std::string &target_server_name, const psi::AlicePbaAndBF &alicePbaAndBF) {
  auto communicator_ptr = reinterpret_cast<AlicePbaAndBFCommunicator *>(communicators_[KAlicePbaAndBF].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  return communicator_ptr->Send(target_server_name, alicePbaAndBF);
}

bool VerticalServer::Send(const std::string &target_server_name, const psi::AliceCheck &aliceCheck) {
  auto communicator_ptr = reinterpret_cast<AliceCheckCommunicator *>(communicators_[KAliceCheck].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  return communicator_ptr->Send(target_server_name, aliceCheck);
}

bool VerticalServer::Send(const std::string &target_server_name, const psi::PlainData &plainData) {
  auto communicator_ptr = reinterpret_cast<PlainDataCommunicator *>(communicators_[KPlainData].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  return communicator_ptr->Send(target_server_name, plainData);
}

WorkerConfigItemPy VerticalServer::Send(const std::string &target_server_name,
                                        const WorkerRegisterItemPy &workerRegisterItem) {
  auto communicator_ptr = reinterpret_cast<DataJoinCommunicator *>(communicators_[KDataJoin].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  return communicator_ptr->Send(target_server_name, workerRegisterItem);
}

void VerticalServer::Receive(const std::string &target_server_name, TensorListItemPy *tensorListItemPy) {
  auto communicator_ptr = reinterpret_cast<TrainerCommunicator *>(communicators_[KTrainer].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  *tensorListItemPy = std::move(communicator_ptr->Receive(target_server_name));
}

void VerticalServer::Receive(const std::string &target_server_name, psi::BobPb *bobPb) {
  auto communicator_ptr = reinterpret_cast<BobPbCommunicator *>(communicators_[KBobPb].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  *bobPb = std::move(communicator_ptr->Receive(target_server_name));
}

void VerticalServer::Receive(const std::string &target_server_name, psi::ClientPSIInit *clientPSIInit) {
  auto communicator_ptr = reinterpret_cast<ClientPSIInitCommunicator *>(communicators_[KClientPSIInit].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  *clientPSIInit = std::move(communicator_ptr->Receive(target_server_name));
}

void VerticalServer::Receive(const std::string &target_server_name, psi::ServerPSIInit *serverPSIInit) {
  auto communicator_ptr = reinterpret_cast<ServerPSIInitCommunicator *>(communicators_[KServerPSIInit].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  *serverPSIInit = std::move(communicator_ptr->Receive(target_server_name));
}

void VerticalServer::Receive(const std::string &target_server_name, psi::BobAlignResult *bobAlignResult) {
  auto communicator_ptr = reinterpret_cast<BobAlignResultCommunicator *>(communicators_[KBobAlignResult].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  *bobAlignResult = std::move(communicator_ptr->Receive(target_server_name));
}

void VerticalServer::Receive(const std::string &target_server_name, psi::AlicePbaAndBF *alicePbaAndBF) {
  auto communicator_ptr = reinterpret_cast<AlicePbaAndBFCommunicator *>(communicators_[KAlicePbaAndBF].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  *alicePbaAndBF = std::move(communicator_ptr->Receive(target_server_name));
}

void VerticalServer::Receive(const std::string &target_server_name, psi::AliceCheck *aliceCheck) {
  auto communicator_ptr = reinterpret_cast<AliceCheckCommunicator *>(communicators_[KAliceCheck].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  *aliceCheck = std::move(communicator_ptr->Receive(target_server_name));
}

void VerticalServer::Receive(const std::string &target_server_name, psi::PlainData *plainData) {
  auto communicator_ptr = reinterpret_cast<PlainDataCommunicator *>(communicators_[KPlainData].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  *plainData = std::move(communicator_ptr->Receive(target_server_name));
}

bool VerticalServer::DataJoinWaitForStart() {
  auto communicator_ptr = reinterpret_cast<DataJoinCommunicator *>(communicators_[KDataJoin].get());
  MS_EXCEPTION_IF_NULL(communicator_ptr);
  bool res = communicator_ptr->waitForRegister();
  if (!res) {
    MS_LOG(EXCEPTION) << "Data join for starting is time out.";
  }
  return true;
}
}  // namespace fl
}  // namespace mindspore
