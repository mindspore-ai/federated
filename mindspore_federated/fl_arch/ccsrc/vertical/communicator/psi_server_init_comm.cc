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

#include "vertical/communicator/psi_server_init_comm.h"
#include <utility>
#include <string>
#include <vector>
#include <memory>

namespace mindspore {
namespace fl {
void ServerPSIInitCommunicator::InitCommunicator(const std::shared_ptr<HttpCommunicator> &http_communicator) {
  if (http_communicator == nullptr) {
    MS_LOG(EXCEPTION) << "Communicators for vertical ServerPSIInit is nullptr.";
  }
  RegisterMsgCallBack(http_communicator, KServerPSIInit);
  InitHttpClient();
  message_queue_ = std::make_shared<MessageQueue<psi::ServerPSIInit>>();
}

bool ServerPSIInitCommunicator::LaunchMsgHandler(const std::shared_ptr<MessageHandler> &message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, false);
  try {
    MS_LOG(INFO) << "Launching psi ServerPSIInit message handler.";
    if (message->data() == nullptr || message->len() == 0) {
      std::string reason = "request data is nullptr or data len is 0.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }

    datajoin::ServerPSIInitProto serverPSIInitProto;
    serverPSIInitProto.ParseFromArray(message->data(), static_cast<int>(message->len()));

    psi::ServerPSIInit serverPSIInit = ParseServerPSIInitProto(serverPSIInitProto);
    if (!VerifyProtoMessage(serverPSIInit)) {
      std::string reason = "Verify serverPSIInitProto failed for vertical psi.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }
    message_queue_->push(serverPSIInit);
    std::string res = std::to_string(ResponseElem::SUCCESS);
    SendResponseMsg(message, res.c_str(), res.size());
    MS_LOG(INFO) << "Launching psi ServerPSIInit message handler successful.";
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Catch exception when handle job vertical psi " << e.what();
    return false;
  }
  return true;
}

bool ServerPSIInitCommunicator::VerifyProtoMessage(const psi::ServerPSIInit &serverPSIInit) { return true; }

bool ServerPSIInitCommunicator::Send(const psi::ServerPSIInit &serverPSIInit) {
  auto client_psi_init_proto_ptr = std::make_shared<datajoin::ServerPSIInitProto>();
  CreateServerPSIInitProto(client_psi_init_proto_ptr.get(), serverPSIInit);
  std::string data = client_psi_init_proto_ptr->SerializeAsString();
  size_t data_size = data.size();
  MS_LOG(INFO) << "Send serverPSIInitProto size is " << data_size;
  return SendMessage(data.c_str(), data_size, KServerPSIInitMsgType);
}

psi::ServerPSIInit ServerPSIInitCommunicator::Receive() {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive ServerPSIInit message.";
  return message_queue_->pop();
}
}  // namespace fl
}  // namespace mindspore
