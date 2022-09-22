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

#include "vertical/communicator/psi_bob_pb_comm.h"
#include <utility>
#include <string>
#include <vector>
#include <memory>

namespace mindspore {
namespace fl {
void BobPbCommunicator::InitCommunicator(const std::shared_ptr<HttpCommunicator> &http_communicator) {
  if (http_communicator == nullptr) {
    MS_LOG(EXCEPTION) << "Communicators for vertical BobPb is nullptr.";
  }
  RegisterMsgCallBack(http_communicator, KBobPb);
  InitHttpClient();
  message_queue_ = std::make_shared<MessageQueue<psi::BobPb>>();
}

bool BobPbCommunicator::LaunchMsgHandler(const std::shared_ptr<MessageHandler> &message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, false);
  try {
    MS_LOG(INFO) << "Launching psi BobPb message handler.";
    if (message->data() == nullptr || message->len() == 0) {
      std::string reason = "request data is nullptr or data len is 0.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }

    datajoin::BobPbProto bobPbProto;
    bobPbProto.ParseFromArray(message->data(), static_cast<int>(message->len()));

    psi::BobPb bobPb = ParseBobPbProto(bobPbProto);
    if (!VerifyProtoMessage(bobPb)) {
      std::string reason = "Verify bob data failed for vertical psi.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }
    message_queue_->push(bobPb);
    std::string res = std::to_string(ResponseElem::SUCCESS);
    SendResponseMsg(message, res.c_str(), res.size());
    MS_LOG(INFO) << "Launching psi BobPb message handler successful.";
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Catch exception when handle job vertical psi " << e.what();
    return false;
  }
  return true;
}

bool BobPbCommunicator::VerifyProtoMessage(const psi::BobPb &bobPb) { return true; }

bool BobPbCommunicator::Send(const psi::BobPb &bob_pb) {
  auto bob_proto_ptr = std::make_shared<datajoin::BobPbProto>();
  CreateBobPbProto(bob_proto_ptr.get(), bob_pb);
  std::string data = bob_proto_ptr->SerializeAsString();
  size_t data_size = data.size();
  return SendMessage(data.c_str(), data_size, KBobPbMsgType);
}

psi::BobPb BobPbCommunicator::Receive() {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive BobPb message.";
  return message_queue_->pop();
}
}  // namespace fl
}  // namespace mindspore
