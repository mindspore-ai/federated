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

#include "vertical/communicator/abstract_communicator.h"
#include "common/communicator/communicator_base.h"
#include "common/communicator/message_handler.h"

namespace mindspore {
namespace fl {
void AbstractCommunicator::RegisterMsgCallBack(const std::shared_ptr<CommunicatorBase> &http_communicator,
                                               const std::string &name) {
  MS_EXCEPTION_IF_NULL(http_communicator);
  MS_LOG(INFO) << "Vertical " << name << " register message callback.";
  http_communicator->RegisterRoundMsgCallback(
    name, [this](const std::shared_ptr<MessageHandler> &message) { LaunchMsgHandler(message); });
}

std::shared_ptr<CommunicatorBase> AbstractCommunicator::CreateHttpCommunicator() {
  auto http_server_address = VFLContext::instance()->http_server_address();
  std::string server_ip;
  uint32_t http_port = 0;
  if (!CommUtil::SplitIpAddress(http_server_address, &server_ip, &http_port)) {
    MS_LOG_EXCEPTION << "The format of http server address '" << http_server_address << "' is invalid";
  }

  return GetOrCreateHttpComm(server_ip, http_port);
}

void AbstractCommunicator::SendResponseMsg(const std::shared_ptr<MessageHandler> &message, const void *data,
                                           size_t len) {
  if (!verifyResponse(message, data, len)) {
    return;
  }
  if (!message->SendResponse(data, len)) {
    MS_LOG(WARNING) << "Sending response failed.";
    return;
  }
}

bool AbstractCommunicator::verifyResponse(const std::shared_ptr<MessageHandler> &message, const void *data,
                                          size_t len) {
  if (message == nullptr) {
    MS_LOG(WARNING) << "The message handler is nullptr.";
    return false;
  }
  if (data == nullptr || len == 0) {
    std::string reason = "The output of the msg is empty.";
    MS_LOG(WARNING) << reason;
    if (!message->SendResponse(reason.c_str(), reason.size())) {
      MS_LOG(WARNING) << "Sending response failed.";
    }
    return false;
  }
  return true;
}
}  // namespace fl
}  // namespace mindspore
