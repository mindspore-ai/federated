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

#include "vertical/communicator/trainer_communicator.h"
#include <utility>
#include <string>
#include <vector>
#include <memory>

#include "vertical/vfl_context.h"
#include "vertical/utils/tensor_utils.h"

namespace mindspore {
namespace fl {
TrainerCommunicator &TrainerCommunicator::GetInstance() {
  static TrainerCommunicator instance;
  return instance;
}

void TrainerCommunicator::InitTrainerConfigs() {
  std::vector<TrainerConfig> trainer_config = {{"trainer"}};
  trainer_config_ = trainer_config;
}

bool TrainerCommunicator::Start(const uint32_t &timeout) {
  try {
    InitTrainerConfigs();
    const auto &http_comm = CreateHttpCommunicator();
    InitTrainerCommunicator(http_comm);
    StartHttpServer();
    InitHttpClient();
    MS_LOG(INFO) << "Trainer communicator started successfully.";
  } catch (const std::exception &e) {
    MS_LOG_WARNING << "Catch exception and begin exit, exception: " << e.what();
    return false;
  }
  return true;
}

bool TrainerCommunicator::Stop() { return true; }

void TrainerCommunicator::InitTrainerCommunicator(const std::shared_ptr<CommunicatorBase> &http_communicator) {
  if (http_communicator == nullptr) {
    MS_LOG(EXCEPTION) << "Communicators for vertical trainer communicator is nullptr.";
  }
  for (const auto &config : trainer_config_) {
    RegisterMsgCallBack(http_communicator, config.name);
  }

  message_queue_ = std::make_shared<MessageQueue<TensorListItemPy>>();
}

bool TrainerCommunicator::LaunchMsgHandler(const std::shared_ptr<MessageHandler> &message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, false);
  try {
    MS_LOG(INFO) << "Launching vertical trainer.";
    if (message->data() == nullptr || message->len() == 0) {
      std::string reason = "request data is nullptr or data len is 0.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }

    TensorListProto tensorListProto;
    if (!tensorListProto.ParseFromArray(message->data(), message->len())) {
      MS_LOG(WARNING) << "Tensor list proto parse from array failed.";
    }

    if (!VerifyTensorListProto(tensorListProto)) {
      std::string reason = "Verify protobuf format failed for TensorListProto.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }

    TensorListItemPy tensorListItemPy = ParseTensorListProto(tensorListProto);
    if (tensorListItemPy.tensors().size() == 0 && tensorListItemPy.tensorListItems().size() == 0) {
      std::string reason = "Parse tensor list data failed for vertical trainer.";
      MS_LOG(WARNING) << reason;
      return false;
    }

    TensorMsgReceiveHandler(tensorListItemPy);
    std::string res = std::to_string(ResponseElem::SUCCESS);
    SendResponseMsg(message, res.c_str(), res.size());
    MS_LOG(INFO) << "Launching vertical trainer successful.";
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Catch exception when handle job vertical trainer " << e.what();
    return false;
  }
  return true;
}

void TrainerCommunicator::InitHttpClient() {
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

bool TrainerCommunicator::Send(const TensorListItemPy &tensorListItemPy) {
  std::shared_ptr<TensorListProto> tensor_list_proto_ptr = std::make_shared<TensorListProto>();
  CreateTensorListProto(tensor_list_proto_ptr.get(), tensorListItemPy);

  std::string data = tensor_list_proto_ptr->SerializeAsString();
  size_t data_size = data.size();

  const std::string msg_type = "/" + trainer_config_[0].name;

  if (data_size == 0) {
    MS_LOG(WARNING) << "Sending request for data size must be > 0";
    return false;
  }
  auto request_track = AddMessageTrack(1, nullptr);
  if (!http_client_->SendMessage(data.c_str(), data_size, request_track, msg_type, HTTP_CONTENT_TYPE_URL_ENCODED)) {
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

bool TrainerCommunicator::VerifyTensorListProto(const TensorListProto &tensorListProto) { return true; }

bool TrainerCommunicator::TensorMsgReceiveHandler(const TensorListItemPy &tensorListItemPy) {
  message_queue_->push(tensorListItemPy);
  MS_LOG(INFO) << "Trainer push tensorListItemPy message success.";
  return true;
}

TensorListItemPy TrainerCommunicator::Receive(const uint32_t &timeout) {
  std::unique_lock<std::mutex> message_lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin receive tensor message.";
  return message_queue_->pop();
}
}  // namespace fl
}  // namespace mindspore
