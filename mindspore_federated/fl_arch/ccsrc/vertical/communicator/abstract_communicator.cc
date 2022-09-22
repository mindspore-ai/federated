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
#include "vertical/common.h"

namespace mindspore {
namespace fl {
std::shared_ptr<HttpCommunicator> AbstractCommunicator::CreateHttpCommunicator() {
  auto http_server_address = VFLContext::instance()->http_server_address();
  std::string server_ip;
  uint32_t http_port = 0;
  if (!CommUtil::SplitIpAddress(http_server_address, &server_ip, &http_port)) {
    MS_LOG_EXCEPTION << "The format of http server address '" << http_server_address << "' is invalid";
  }
  MS_LOG(INFO) << "Create Http communicator.";
  auto http_server = std::make_shared<HttpServer>(server_ip, http_port, kThreadNum);
  MS_EXCEPTION_IF_NULL(http_server);
  auto http_communicator = std::make_shared<HttpCommunicator>(http_server);
  MS_EXCEPTION_IF_NULL(http_communicator);
  return http_communicator;
}

void AbstractCommunicator::StartHttpServer(const std::shared_ptr<HttpCommunicator> &http_communicator) {
  if (http_communicator == nullptr) {
    MS_LOG(EXCEPTION) << "Http server starting failed.";
  }
  http_communicator->Start();
  auto http_server = http_communicator->http_server();
  MS_LOG(INFO) << "Initialize http server IP:" << http_server->address() << ", PORT:" << http_server->port();
  if (!http_server->Start()) {
    MS_LOG(EXCEPTION) << "Http server starting failed.";
  }
  MS_LOG(INFO) << "Http communicator starte successfully.";
}

void AbstractCommunicator::RegisterMsgCallBack(const std::shared_ptr<HttpCommunicator> &http_communicator,
                                               const std::string &name) {
  MS_EXCEPTION_IF_NULL(http_communicator);
  MS_LOG(INFO) << "Vertical " << name << " register message callback.";
  http_communicator->RegisterRoundMsgCallback(
    name, [this](const std::shared_ptr<MessageHandler> &message) { LaunchMsgHandler(message); });
}

void AbstractCommunicator::InitHttpClient() {
  remote_server_address_ = VFLContext::instance()->remote_server_address();
  if (VFLContext::instance()->enable_ssl()) {
    remote_server_address_ = "https://" + remote_server_address_;
  } else {
    remote_server_address_ = "http://" + remote_server_address_;
  }

  MS_LOG(INFO) << "Request will be sent to server domain:" << remote_server_address_;
  http_client_ = std::make_shared<HttpClient>(remote_server_address_);

  http_client_->SetMessageCallback([&](const std::shared_ptr<ResponseTrack> &response_track,
                                       const std::string &msg_type) { NotifyMessageArrival(response_track); });
  http_client_->Init();
}

bool AbstractCommunicator::SendMessage(const void *data, size_t data_size, const std::string &msg_type) {
  if (data == nullptr) {
    MS_LOG(WARNING) << "Data for sending request is nullptr.";
    return false;
  }
  if (data_size == 0) {
    MS_LOG(WARNING) << "Data size for sending request must be > 0";
    return false;
  }
  auto request_track = AddMessageTrack(1, nullptr);
  if (!http_client_->SendMessage(data, data_size, request_track, msg_type, HTTP_CONTENT_TYPE_URL_ENCODED)) {
    MS_LOG(WARNING) << "Sending request for msg type:" << msg_type << " to server " << remote_server_address_
                    << " failed.";
    return false;
  }
  if (!Wait(request_track)) {
    MS_LOG(WARNING) << "Sending http message timeout.";
    http_client_->BreakLoopEvent();
    return false;
  }
  std::string res_msg = reinterpret_cast<char *>(http_client_->response_msg()->data());
  return res_msg == std::to_string(ResponseElem::SUCCESS);
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
