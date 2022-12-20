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
      auto queue = std::make_shared<MessageQueue<SliceProto>>();
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
    std::string message_offset = message->message_offset();
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

    MS_LOG(INFO) << "Request source server name is " << message_source << ", message type is " << message_type
                 << ", message offset is " << message_offset;
    auto msg_data = static_cast<uint8_t *>(const_cast<void *>(message->data()));
    std::vector<uint8_t> slice_data{msg_data, msg_data + message->len()};
    SliceProto slice_proto = {slice_data, message_offset};
    auto queue = message_queues_[message_source][message_type];
    MS_EXCEPTION_IF_NULL(queue);
    queue->push(slice_proto);

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
  auto slice_proto = CreateProtoWithSlices(aliceCheck);
  auto slice_data = slice_proto.slice_data;
  auto offset = slice_proto.offset;
  auto response_msg =
    SendMessage(target_server_name, slice_data.data(), slice_data.size(), KPsiUri, KAliceCheck, offset);
  return ResponseHandler(response_msg);
}

bool PsiCommunicator::Send(const std::string &target_server_name, const psi::BobPb &bob_pb) {
  auto slice_proto = CreateProtoWithSlices(bob_pb);
  auto slice_data = slice_proto.slice_data;
  auto offset = slice_proto.offset;
  auto response_msg = SendMessage(target_server_name, slice_data.data(), slice_data.size(), KPsiUri, KBobPb, offset);
  return ResponseHandler(response_msg);
}

bool PsiCommunicator::Send(const std::string &target_server_name, const psi::AlicePbaAndBF &alicePbaAndBF) {
  auto slice_proto = CreateProtoWithSlices(alicePbaAndBF);
  auto slice_data = slice_proto.slice_data;
  auto offset = slice_proto.offset;
  auto response_msg =
    SendMessage(target_server_name, slice_data.data(), slice_data.size(), KPsiUri, KAlicePbaAndBF, offset);
  return ResponseHandler(response_msg);
}

bool PsiCommunicator::Send(const std::string &target_server_name, const psi::BobAlignResult &bobAlignResult) {
  auto slice_proto = CreateProtoWithSlices(bobAlignResult);
  auto slice_data = slice_proto.slice_data;
  auto offset = slice_proto.offset;
  auto response_msg =
    SendMessage(target_server_name, slice_data.data(), slice_data.size(), KPsiUri, KBobAlignResult, offset);
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
  auto slice_proto = CreateProtoWithSlices(plain_data);
  auto slice_data = slice_proto.slice_data;
  auto offset = slice_proto.offset;
  auto response_msg =
    SendMessage(target_server_name, slice_data.data(), slice_data.size(), KPsiUri, KPlainData, offset);
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
  auto slice_proto = queue->pop(kPsiWaitSecondTimes);
  auto slice_data = slice_proto.slice_data;
  auto offset = slice_proto.offset;

  std::vector<datajoin::AliceCheckProto> protos;
  std::vector<std::string> offset_vec = StringSplit(offset, KProtoSplitSign);
  auto begin_ptr = slice_data.data();
  size_t index = 0;
  for (const auto &item : offset_vec) {
    size_t size = std::stoull(item.c_str()) - index;
    datajoin::AliceCheckProto proto;
    proto.ParseFromArray(begin_ptr, size);
    protos.emplace_back(proto);
    begin_ptr += size;
    index += size;
  }
  *aliceCheck = std::move(ParseProtoWithSlices(protos));
}

void PsiCommunicator::Receive(const std::string &target_server_name, psi::BobPb *bobPb) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive BobPb message.";
  if (message_queues_.count(target_server_name) == 0) {
    MS_LOG(EXCEPTION) << "Target server name " << target_server_name << " for message queues is invalid.";
  }
  auto queue = message_queues_[target_server_name][KBobPb];
  MS_EXCEPTION_IF_NULL(queue);
  auto slice_proto = queue->pop(kPsiWaitSecondTimes);
  auto slice_data = slice_proto.slice_data;
  auto offset = slice_proto.offset;
  std::vector<datajoin::BobPbProto> protos;
  std::vector<std::string> offset_vec = StringSplit(offset, KProtoSplitSign);
  auto begin_ptr = slice_data.data();
  size_t index = 0;
  for (const auto &item : offset_vec) {
    size_t size = std::stoull(item.c_str()) - index;
    datajoin::BobPbProto proto;
    proto.ParseFromArray(begin_ptr, size);
    protos.emplace_back(proto);
    begin_ptr += size;
    index += size;
  }

  *bobPb = std::move(ParseProtoWithSlices(protos));
}

void PsiCommunicator::Receive(const std::string &target_server_name, psi::AlicePbaAndBF *alicePbaAndBF) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive AlicePbaAndBF message.";
  if (message_queues_.count(target_server_name) == 0) {
    MS_LOG(EXCEPTION) << "Target server name " << target_server_name << " for message queues is invalid.";
  }
  auto queue = message_queues_[target_server_name][KAlicePbaAndBF];
  MS_EXCEPTION_IF_NULL(queue);
  auto slice_proto = queue->pop(kPsiWaitSecondTimes);
  auto slice_data = slice_proto.slice_data;
  auto offset = slice_proto.offset;

  std::vector<datajoin::AlicePbaAndBFProto> protos;
  std::vector<std::string> offset_vec = StringSplit(offset, KProtoSplitSign);
  auto begin_ptr = slice_data.data();
  size_t index = 0;
  for (const auto &item : offset_vec) {
    size_t size = std::stoull(item.c_str()) - index;
    datajoin::AlicePbaAndBFProto proto;
    proto.ParseFromArray(begin_ptr, size);
    protos.emplace_back(proto);
    begin_ptr += size;
    index += size;
  }
  *alicePbaAndBF = std::move(ParseProtoWithSlices(protos));
}

void PsiCommunicator::Receive(const std::string &target_server_name, psi::BobAlignResult *bobAlignResult) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive BobAlignResult message.";
  if (message_queues_.count(target_server_name) == 0) {
    MS_LOG(EXCEPTION) << "Target server name " << target_server_name << " for message queues is invalid.";
  }
  auto queue = message_queues_[target_server_name][KBobAlignResult];
  MS_EXCEPTION_IF_NULL(queue);
  auto slice_proto = queue->pop(kPsiWaitSecondTimes);
  auto slice_data = slice_proto.slice_data;
  auto offset = slice_proto.offset;

  std::vector<datajoin::BobAlignResultProto> protos;
  std::vector<std::string> offset_vec = StringSplit(offset, KProtoSplitSign);
  auto begin_ptr = slice_data.data();
  size_t index = 0;
  for (const auto &item : offset_vec) {
    size_t size = std::stoull(item.c_str()) - index;
    datajoin::BobAlignResultProto proto;
    proto.ParseFromArray(begin_ptr, size);
    protos.emplace_back(proto);
    begin_ptr += size;
    index += size;
  }
  *bobAlignResult = std::move(ParseProtoWithSlices(protos));
}

void PsiCommunicator::Receive(const std::string &target_server_name, psi::ClientPSIInit *clientPSIInit) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive ClientPSIInit message.";
  if (message_queues_.count(target_server_name) == 0) {
    MS_LOG(EXCEPTION) << "Target server name " << target_server_name << " for message queues is invalid.";
  }
  auto queue = message_queues_[target_server_name][KClientPSIInit];
  MS_EXCEPTION_IF_NULL(queue);
  auto slice_proto = queue->pop(kPsiWaitSecondTimes);
  auto slice_data = slice_proto.slice_data;
  datajoin::ClientPSIInitProto proto;
  proto.ParseFromArray(slice_data.data(), static_cast<int>(slice_data.size()));

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
  auto slice_proto = queue->pop(kPsiWaitSecondTimes);
  auto slice_data = slice_proto.slice_data;
  auto offset = slice_proto.offset;

  std::vector<datajoin::PlainDataProto> protos;
  std::vector<std::string> offset_vec = StringSplit(offset, KProtoSplitSign);
  auto begin_ptr = slice_data.data();
  size_t index = 0;
  for (const auto &item : offset_vec) {
    size_t size = std::stoull(item.c_str()) - index;
    datajoin::PlainDataProto proto;
    proto.ParseFromArray(begin_ptr, size);
    protos.emplace_back(proto);
    begin_ptr += size;
    index += size;
  }
  *plainData = std::move(ParseProtoWithSlices(protos));
}

void PsiCommunicator::Receive(const std::string &target_server_name, psi::ServerPSIInit *serverPSIInit) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive ServerPSIInit message.";
  if (message_queues_.count(target_server_name) == 0) {
    MS_LOG(EXCEPTION) << "Target server name " << target_server_name << " for message queues is invalid.";
  }
  auto queue = message_queues_[target_server_name][KServerPSIInit];
  MS_EXCEPTION_IF_NULL(queue);
  auto slice_proto = queue->pop(kPsiWaitSecondTimes);
  auto slice_data = slice_proto.slice_data;

  datajoin::ServerPSIInitProto proto;
  proto.ParseFromArray(slice_data.data(), static_cast<int>(slice_data.size()));

  *serverPSIInit = std::move(ParseServerPSIInitProto(proto));
}
}  // namespace fl
}  // namespace mindspore
