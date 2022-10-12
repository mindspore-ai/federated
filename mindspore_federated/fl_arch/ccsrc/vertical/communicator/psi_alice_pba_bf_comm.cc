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

#include "vertical/communicator/psi_alice_pba_bf_comm.h"
#include <utility>
#include <string>
#include <vector>
#include <memory>

namespace mindspore {
namespace fl {
void AlicePbaAndBFCommunicator::InitCommunicator(const std::shared_ptr<HttpCommunicator> &http_communicator) {
  if (http_communicator == nullptr) {
    MS_LOG(EXCEPTION) << "Communicators for vertical AlicePbaAndBF is nullptr.";
  }
  RegisterMsgCallBack(http_communicator, KAlicePbaAndBF);
  InitHttpClient();
  auto remote_server_address = VFLContext::instance()->remote_server_address();
  for (const auto &item : remote_server_address) {
    auto target_server_name = item.first;
    auto queue = std::make_shared<MessageQueue<psi::AlicePbaAndBF>>();
    message_queues_[target_server_name] = queue;
  }
}

bool AlicePbaAndBFCommunicator::LaunchMsgHandler(const std::shared_ptr<MessageHandler> &message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, false);
  try {
    MS_LOG(INFO) << "Launching psi AlicePbaAndBF message handler.";
    if (message->data() == nullptr || message->len() == 0) {
      std::string reason = "request data is nullptr or data len is 0.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }
    std::string message_type = message->message_type();
    if (message_type.empty() || message_queues_.count(message_type) == 0) {
      std::string reason = "Request message type is invalid.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }
    MS_LOG(INFO) << "Request message type is " << message_type;

    datajoin::AlicePbaAndBFProto alicePbaAndBFProto;
    alicePbaAndBFProto.ParseFromArray(message->data(), static_cast<int>(message->len()));

    psi::AlicePbaAndBF alicePbaAndBF = ParseAlicePbaAndBFProto(alicePbaAndBFProto);
    if (!VerifyProtoMessage(alicePbaAndBF)) {
      std::string reason = "Verify bob data failed for vertical psi.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }
    auto queue = message_queues_[message_type];
    MS_EXCEPTION_IF_NULL(queue);
    queue->push(alicePbaAndBF);
    std::string res = std::to_string(ResponseElem::SUCCESS);
    SendResponseMsg(message, res.c_str(), res.size());
    MS_LOG(INFO) << "Launching psi BobAlignResult message handler successful.";
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Catch exception when handle job vertical psi " << e.what();
    return false;
  }
  return true;
}

bool AlicePbaAndBFCommunicator::VerifyProtoMessage(const psi::AlicePbaAndBF &alicePbaAndBF) { return true; }

bool AlicePbaAndBFCommunicator::Send(const std::string &target_server_name, const psi::AlicePbaAndBF &alicePbaAndBF) {
  auto alice_pba_bf_proto_ptr = std::make_shared<datajoin::AlicePbaAndBFProto>();
  CreateAlicePbaAndBFProto(alice_pba_bf_proto_ptr.get(), alicePbaAndBF);
  std::string data = alice_pba_bf_proto_ptr->SerializeAsString();
  size_t data_size = data.size();
  auto response_msg = SendMessage(target_server_name, data.c_str(), data_size, KAlicePbaAndBFMsgType);
  std::string response_data = response_msg == nullptr ? "" : reinterpret_cast<char *>(response_msg->data());
  return response_data == std::to_string(ResponseElem::SUCCESS);
}

psi::AlicePbaAndBF AlicePbaAndBFCommunicator::Receive(const std::string &target_server_name) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive AlicePbaAndBF message.";
  if (message_queues_.count(target_server_name) == 0) {
    MS_LOG(EXCEPTION) << "Target server name " << target_server_name << " for message queues is invalid.";
  }
  auto queue = message_queues_[target_server_name];
  MS_EXCEPTION_IF_NULL(queue);
  return queue->pop();
}
}  // namespace fl
}  // namespace mindspore
