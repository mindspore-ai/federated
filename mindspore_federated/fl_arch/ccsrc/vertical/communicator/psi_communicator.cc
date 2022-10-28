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

#include "vertical/communicator/psi_communicator.h"
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

namespace mindspore {
namespace fl {
void PsiCommunicator::InitCommunicator(const std::shared_ptr<HttpCommunicator> &http_communicator) {
  if (http_communicator == nullptr) {
    MS_LOG(EXCEPTION) << "Communicators for vertical AliceCheck is nullptr.";
  }
  RegisterMsgCallBack(http_communicator, KPsi);
  InitHttpClient();

  std::vector<VerticalConfig> psi_config = {{KBobPb},          {KClientPSIInit}, {KServerPSIInit}, {KAlicePbaAndBF},
                                            {KBobAlignResult}, {KAliceCheck},    {KPlainData}};
  auto remote_server_address = VFLContext::instance()->remote_server_address();
  for (const auto &address : remote_server_address) {
    std::map<std::string, MessageQueuePtr> qmap;
    auto target_server_name = address.first;
    for (const auto &config : psi_config) {
      auto queue = std::make_shared<MessageQueue<std::vector<uint8_t>>>();
      qmap[config.name] = queue;
    }
    message_queues_[target_server_name] = qmap;
  }
}

bool PsiCommunicator::VerifyProtoMessage(const psi::PlainData &plain_data) { return true; }

bool PsiCommunicator::LaunchMsgHandler(const std::shared_ptr<MessageHandler> &message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, false);
  try {
    MS_LOG(INFO) << "Launching psi message handler.";
    if (message->data() == nullptr || message->len() == 0) {
      std::string reason = "Request data is nullptr or data len is 0.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }
    std::string message_source = message->message_source();
    std::string message_type = message->message_type();
    if (message_queues_.count(message_source) <= 0) {
      std::string reason = "Request message source server name " + message_source + " is invalid.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }

    if (message_queues_[message_source].count(message_type) <= 0) {
      std::string reason = "Request message type " + message_type + " is invalid.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }

    MS_LOG(INFO) << "Request source server name is " << message_source << ", message type is " << message_type;
    auto msg_data = static_cast<uint8_t *>(const_cast<void *>(message->data()));
    std::vector<uint8_t> data{msg_data, msg_data + message->len()};

    auto queue = message_queues_[message_source][message_type];
    MS_EXCEPTION_IF_NULL(queue);
    queue->push(data);

    std::string res = toString(ResponseElem::SUCCESS);
    SendResponseMsg(message, res.c_str(), res.size());
    MS_LOG(INFO) << "Launching psi message handler successful. Response msg is " << res;
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Catch exception when handle job vertical psi " << e.what();
    return false;
  }
  return true;
}

bool PsiCommunicator::ResponseHandler(const std::shared_ptr<std::vector<uint8_t>> &response_msg) {
  if (!response_msg) {
    return false;
  }
  return reinterpret_cast<char *>(response_msg->data()) == toString(ResponseElem::SUCCESS);
}

bool PsiCommunicator::Send(const std::string &target_server_name, const psi::AliceCheck &aliceCheck) {
  auto alice_check_proto_ptr = std::make_shared<datajoin::AliceCheckProto>();
  CreateAliceCheckProto(alice_check_proto_ptr.get(), aliceCheck);
  std::string data = alice_check_proto_ptr->SerializeAsString();
  size_t data_size = data.size();
  auto response_msg = SendMessage(target_server_name, data.c_str(), data_size, KPsiUri, KAliceCheck);
  return ResponseHandler(response_msg);
}

bool PsiCommunicator::Send(const std::string &target_server_name, const psi::BobPb &bob_pb) {
  auto bob_proto_ptr = std::make_shared<datajoin::BobPbProto>();
  CreateBobPbProto(bob_proto_ptr.get(), bob_pb);
  std::string data = bob_proto_ptr->SerializeAsString();
  size_t data_size = data.size();
  auto response_msg = SendMessage(target_server_name, data.c_str(), data_size, KPsiUri, KBobPb);
  return ResponseHandler(response_msg);
}

bool PsiCommunicator::Send(const std::string &target_server_name, const psi::AlicePbaAndBF &alicePbaAndBF) {
  auto alice_pba_bf_proto_ptr = std::make_shared<datajoin::AlicePbaAndBFProto>();
  CreateAlicePbaAndBFProto(alice_pba_bf_proto_ptr.get(), alicePbaAndBF);
  std::string data = alice_pba_bf_proto_ptr->SerializeAsString();
  size_t data_size = data.size();
  auto response_msg = SendMessage(target_server_name, data.c_str(), data_size, KPsiUri, KAlicePbaAndBF);
  return ResponseHandler(response_msg);
}

bool PsiCommunicator::Send(const std::string &target_server_name, const psi::BobAlignResult &bobAlignResult) {
  auto bob_align_result_proto_ptr = std::make_shared<datajoin::BobAlignResultProto>();
  CreateBobAlignResultProto(bob_align_result_proto_ptr.get(), bobAlignResult);
  std::string data = bob_align_result_proto_ptr->SerializeAsString();
  size_t data_size = data.size();
  auto response_msg = SendMessage(target_server_name, data.c_str(), data_size, KPsiUri, KBobAlignResult);
  return ResponseHandler(response_msg);
}

bool PsiCommunicator::Send(const std::string &target_server_name, const psi::ClientPSIInit &clientPSIInit) {
  auto client_psi_init_proto_ptr = std::make_shared<datajoin::ClientPSIInitProto>();
  CreateClientPSIInitProto(client_psi_init_proto_ptr.get(), clientPSIInit);
  std::string data = client_psi_init_proto_ptr->SerializeAsString();
  size_t data_size = data.size();
  MS_LOG(INFO) << "Send clientPSIInitProto size is " << data_size;
  auto response_msg = SendMessage(target_server_name, data.c_str(), data_size, KPsiUri, KClientPSIInit);
  return ResponseHandler(response_msg);
}

bool PsiCommunicator::Send(const std::string &target_server_name, const psi::PlainData &plain_data) {
  auto plain_proto_ptr = std::make_shared<datajoin::PlainDataProto>();
  CreatePlainDataProto(plain_proto_ptr.get(), plain_data);
  std::string data = plain_proto_ptr->SerializeAsString();
  size_t data_size = data.size();
  auto response_msg = SendMessage(target_server_name, data.c_str(), data_size, KPsiUri, KPlainData);
  return ResponseHandler(response_msg);
}

bool PsiCommunicator::Send(const std::string &target_server_name, const psi::ServerPSIInit &serverPSIInit) {
  auto client_psi_init_proto_ptr = std::make_shared<datajoin::ServerPSIInitProto>();
  CreateServerPSIInitProto(client_psi_init_proto_ptr.get(), serverPSIInit);
  std::string data = client_psi_init_proto_ptr->SerializeAsString();
  size_t data_size = data.size();
  MS_LOG(INFO) << "Send serverPSIInitProto size is " << data_size;
  auto response_msg = SendMessage(target_server_name, data.c_str(), data_size, KPsiUri, KServerPSIInit);
  return ResponseHandler(response_msg);
}

void PsiCommunicator::Receive(const std::string &target_server_name, psi::AliceCheck *aliceCheck) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive AliceCheck message.";

  if (message_queues_.count(target_server_name) == 0) {
    MS_LOG(EXCEPTION) << "Target server name " << target_server_name << " for message queues is invalid.";
  }
  auto queue = message_queues_[target_server_name][KAliceCheck];
  MS_EXCEPTION_IF_NULL(queue);
  auto vec_data = queue->pop(kPsiWaitSecondTimes);
  datajoin::AliceCheckProto proto;
  proto.ParseFromArray(vec_data.data(), static_cast<int>(vec_data.size()));
  *aliceCheck = std::move(ParseAliceCheckProto(proto));
}

void PsiCommunicator::Receive(const std::string &target_server_name, psi::BobPb *bobPb) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive BobPb message.";
  if (message_queues_.count(target_server_name) == 0) {
    MS_LOG(EXCEPTION) << "Target server name " << target_server_name << " for message queues is invalid.";
  }
  auto queue = message_queues_[target_server_name][KBobPb];
  MS_EXCEPTION_IF_NULL(queue);
  auto vec_data = queue->pop(kPsiWaitSecondTimes);

  datajoin::BobPbProto proto;
  proto.ParseFromArray(vec_data.data(), static_cast<int>(vec_data.size()));

  *bobPb = std::move(ParseBobPbProto(proto));
}

void PsiCommunicator::Receive(const std::string &target_server_name, psi::AlicePbaAndBF *alicePbaAndBF) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive AlicePbaAndBF message.";
  if (message_queues_.count(target_server_name) == 0) {
    MS_LOG(EXCEPTION) << "Target server name " << target_server_name << " for message queues is invalid.";
  }
  auto queue = message_queues_[target_server_name][KAlicePbaAndBF];
  MS_EXCEPTION_IF_NULL(queue);
  auto vec_data = queue->pop(kPsiWaitSecondTimes);

  datajoin::AlicePbaAndBFProto proto;
  proto.ParseFromArray(vec_data.data(), static_cast<int>(vec_data.size()));

  *alicePbaAndBF = std::move(ParseAlicePbaAndBFProto(proto));
}

void PsiCommunicator::Receive(const std::string &target_server_name, psi::BobAlignResult *bobAlignResult) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive BobAlignResult message.";
  if (message_queues_.count(target_server_name) == 0) {
    MS_LOG(EXCEPTION) << "Target server name " << target_server_name << " for message queues is invalid.";
  }
  auto queue = message_queues_[target_server_name][KBobAlignResult];
  MS_EXCEPTION_IF_NULL(queue);
  auto vec_data = queue->pop(kPsiWaitSecondTimes);
  datajoin::BobAlignResultProto proto;
  proto.ParseFromArray(vec_data.data(), static_cast<int>(vec_data.size()));

  *bobAlignResult = std::move(ParseBobAlignResultProto(proto));
}

void PsiCommunicator::Receive(const std::string &target_server_name, psi::ClientPSIInit *clientPSIInit) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive ClientPSIInit message.";
  if (message_queues_.count(target_server_name) == 0) {
    MS_LOG(EXCEPTION) << "Target server name " << target_server_name << " for message queues is invalid.";
  }
  auto queue = message_queues_[target_server_name][KClientPSIInit];
  MS_EXCEPTION_IF_NULL(queue);
  auto vec_data = queue->pop(kPsiWaitSecondTimes);

  datajoin::ClientPSIInitProto proto;
  proto.ParseFromArray(vec_data.data(), static_cast<int>(vec_data.size()));

  *clientPSIInit = std::move(ParseClientPSIInitProto(proto));
}

void PsiCommunicator::Receive(const std::string &target_server_name, psi::PlainData *plainData) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive PlainData message.";
  if (message_queues_.count(target_server_name) == 0) {
    MS_LOG(EXCEPTION) << "Target server name " << target_server_name << " for message queues is invalid.";
  }
  auto queue = message_queues_[target_server_name][KPlainData];
  MS_EXCEPTION_IF_NULL(queue);
  auto vec_data = queue->pop(kPsiWaitSecondTimes);

  datajoin::PlainDataProto proto;
  proto.ParseFromArray(vec_data.data(), static_cast<int>(vec_data.size()));
  *plainData = std::move(ParsePlainDataProto(proto));
}

void PsiCommunicator::Receive(const std::string &target_server_name, psi::ServerPSIInit *serverPSIInit) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive ServerPSIInit message.";
  if (message_queues_.count(target_server_name) == 0) {
    MS_LOG(EXCEPTION) << "Target server name " << target_server_name << " for message queues is invalid.";
  }
  auto queue = message_queues_[target_server_name][KServerPSIInit];
  MS_EXCEPTION_IF_NULL(queue);
  auto vec_data = queue->pop(kPsiWaitSecondTimes);

  datajoin::ServerPSIInitProto proto;
  proto.ParseFromArray(vec_data.data(), static_cast<int>(vec_data.size()));

  *serverPSIInit = std::move(ParseServerPSIInitProto(proto));
}
}  // namespace fl
}  // namespace mindspore
