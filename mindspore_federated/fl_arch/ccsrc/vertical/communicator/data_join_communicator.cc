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

#include "vertical/communicator/data_join_communicator.h"
#include <utility>
#include <string>
#include <vector>
#include <memory>

namespace mindspore {
namespace fl {
void DataJoinCommunicator::InitCommunicator(const std::shared_ptr<HttpCommunicator> &http_communicator) {
  if (http_communicator == nullptr) {
    MS_LOG(EXCEPTION) << "Communicators for vertical ClientPSIInit is nullptr.";
  }
  RegisterMsgCallBack(http_communicator, KDataJoin);
  InitHttpClient();
}

bool DataJoinCommunicator::LaunchMsgHandler(const std::shared_ptr<MessageHandler> &message) {
  MS_ERROR_IF_NULL_W_RET_VAL(message, false);
  try {
    MS_LOG(INFO) << "Launching data join data join message handler.";
    if (message->data() == nullptr || message->len() == 0) {
      std::string reason = "request data is nullptr or data len is 0.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }
    datajoin::WorkerRegisterProto proto;
    proto.ParseFromArray(message->data(), static_cast<int>(message->len()));

    WorkerRegisterItemPy workerRegisterItemPy = ParseWorkerRegisterProto(proto);
    if (!VerifyProtoMessage(workerRegisterItemPy)) {
      std::string reason = "Verify clientPSIInitProto failed for vertical psi.";
      MS_LOG(WARNING) << reason;
      SendResponseMsg(message, reason.c_str(), reason.size());
      return false;
    }
    MS_LOG(INFO) << "Worker register name is " << workerRegisterItemPy.worker_name();
    notifyForRegister();
    SendWorkerConfig(message);
    MS_LOG(INFO) << "Launching data join message handler successful.";
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Catch exception when handle job vertical psi " << e.what();
    return false;
  }
  return true;
}

bool DataJoinCommunicator::VerifyProtoMessage(const WorkerRegisterItemPy &workerRegisterItemPy) { return true; }

WorkerConfigItemPy DataJoinCommunicator::Send(const std::string &target_server_name,
                                              const WorkerRegisterItemPy &workerRegisterItemPy) {
  auto worker_register_proto_ptr = std::make_shared<datajoin::WorkerRegisterProto>();
  CreateWorkerRegisterProto(worker_register_proto_ptr.get(), workerRegisterItemPy);
  std::string data = worker_register_proto_ptr->SerializeAsString();
  size_t data_size = data.size();
  MS_LOG(INFO) << "Send WorkerRegisterProto size is " << data_size;

  WorkerConfigItemPy workerConfigItemPy;
  auto response_msg = SendMessage(target_server_name, data.c_str(), data_size, KDataJoinUri, KDataJoin);
  if (response_msg == nullptr) {
    return workerConfigItemPy;
  }
  datajoin::WorkerConfigProto proto;
  proto.ParseFromArray(response_msg->data(), static_cast<int>(response_msg->size()));
  workerConfigItemPy = ParseWorkerConfigProto(proto);
  return workerConfigItemPy;
}

void DataJoinCommunicator::notifyForRegister() {
  is_worker_registered_ = true;
  message_received_cond_.notify_all();
}

void DataJoinCommunicator::SendWorkerConfig(const std::shared_ptr<MessageHandler> &message) {
  auto worker_config_proto_ptr = std::make_shared<datajoin::WorkerConfigProto>();
  auto worker_config = VFLContext::instance()->worker_config();
  CreateWorkerConfigProto(worker_config_proto_ptr.get(), worker_config);
  std::string data = worker_config_proto_ptr->SerializeAsString();
  size_t data_size = data.size();
  SendResponseMsg(message, data.c_str(), data_size);
}

bool DataJoinCommunicator::waitForRegister(const uint32_t &timeout) {
  std::unique_lock<std::mutex> lock(message_received_mutex_);
  MS_LOG(INFO) << "Begin wait for client worker register.";
  for (uint32_t i = 0; i < timeout; i++) {
    bool res = message_received_cond_.wait_for(lock, std::chrono::seconds(1), [this] { return is_worker_registered_; });
    if (res) {
      return res;
    }
  }
  return false;
}
}  // namespace fl
}  // namespace mindspore
