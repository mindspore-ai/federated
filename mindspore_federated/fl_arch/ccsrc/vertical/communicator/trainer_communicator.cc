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
void TrainerCommunicator::InitCommunicator(const std::shared_ptr<HttpCommunicator> &http_communicator) {
  if (http_communicator == nullptr) {
    MS_LOG(EXCEPTION) << "Communicators for vertical trainer communicator is nullptr.";
  }
  RegisterMsgCallBack(http_communicator, KTrainer);
  InitHttpClient();
  message_queue_ = std::make_shared<MessageQueue<TensorListItemPy>>();
}

bool TrainerCommunicator::VerifyTensorListItem(const TensorListItemPy &tensorListItemPy) {
  if (tensorListItemPy.tensors().size() == 0 && tensorListItemPy.tensorListItems().size() == 0) {
    return false;
  }
  return true;
}

bool TrainerCommunicator::LaunchMsgHandler(const std::shared_ptr<MessageHandler> &message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, false);
  try {
    MS_LOG(INFO) << "Launching vertical trainer message handler.";
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

    TensorListItemPy tensorListItemPy = ParseTensorListProto(tensorListProto);
    if (!VerifyTensorListItem(tensorListItemPy)) {
      std::string reason = "Verify tensor list data failed for vertical trainer.";
      SendResponseMsg(message, reason.c_str(), reason.size());
      MS_LOG(WARNING) << reason;
      return false;
    }

    TensorMsgReceiveHandler(tensorListItemPy);
    std::string res = std::to_string(ResponseElem::SUCCESS);
    SendResponseMsg(message, res.c_str(), res.size());
    MS_LOG(INFO) << "Launching vertical trainer message handler successful.";
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Catch exception when handle job vertical trainer " << e.what();
    return false;
  }
  return true;
}

bool TrainerCommunicator::Send(const TensorListItemPy &tensorListItemPy) {
  std::shared_ptr<TensorListProto> tensor_list_proto_ptr = std::make_shared<TensorListProto>();
  CreateTensorListProto(tensor_list_proto_ptr.get(), tensorListItemPy);
  std::string data = tensor_list_proto_ptr->SerializeAsString();
  size_t data_size = data.size();
  return SendMessage(data.c_str(), data_size, KTrainerMsgType);
}

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
