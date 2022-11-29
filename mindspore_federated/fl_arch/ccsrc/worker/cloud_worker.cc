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
#include "worker/cloud_worker.h"

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "worker/worker_node.h"
#include "armour/secure_protocol/key_agreement.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace fl {
namespace worker {
CloudWorker &CloudWorker::GetInstance() {
  static CloudWorker instance;
  return instance;
}

void CloudWorker::Init() {
  if (running_.load()) {
    return;
  }
  running_ = true;
  MS_LOG(INFO) << "Begin to run federated learning cloud worker.";

  worker_node_ = std::make_shared<fl::WorkerNode>();
  MS_EXCEPTION_IF_NULL(worker_node_);

  if (!worker_node_->Start()) {
    MS_LOG(EXCEPTION) << "Starting worker node failed.";
  }
  fl_name_ = FLContext::instance()->fl_name();

  http_server_address_ = FLContext::instance()->http_server_address();
  if (FLContext::instance()->enable_ssl()) {
    http_server_address_ = "https://" + http_server_address_;
  } else {
    http_server_address_ = "http://" + http_server_address_;
  }

  MS_LOG(INFO) << "fl id is:" << fl_id() << ". Request will be sent to server domain:" << http_server_address_;
  http_client_ = std::make_shared<HttpClient>(http_server_address_);

  http_client_->SetMessageCallback([&](const std::shared_ptr<ResponseTrack> &response_track,
                                       const std::string &msg_type) { NotifyMessageArrival(response_track); });

  http_client_->Init();
}

std::shared_ptr<std::vector<unsigned char>> CloudWorker::SendToServerSync(const void *data, size_t data_size,
                                                                          const std::string &msg_type,
                                                                          const std::string &content_type) {
  MS_ERROR_IF_NULL_W_RET_VAL(data, nullptr);
  if (data_size == 0) {
    MS_LOG(WARNING) << "Sending request for data size must be > 0";
    return nullptr;
  }
  auto request_track = AddMessageTrack(1, nullptr);
  if (!http_client_->SendMessage(data, data_size, request_track, msg_type, content_type)) {
    MS_LOG(WARNING) << "Sending request for msg type:" << msg_type << " to server " << http_server_address_
                    << " failed.";
    return nullptr;
  }
  if (!Wait(request_track)) {
    MS_LOG(WARNING) << "Sending http message timeout.";
    http_client_->BreakLoopEvent();
    return nullptr;
  }
  return http_client_->response_msg();
}

void CloudWorker::RegisterMessageCallback(const std::string msg_type, const MessageReceive &cb) {
  if (handlers_.count(msg_type) > 0) {
    MS_LOG(DEBUG) << "Http handlers has already register msg type:" << msg_type;
    return;
  }
  handlers_[msg_type] = cb;
  MS_LOG(INFO) << "Http handlers register msg type:" << msg_type;
}

void CloudWorker::set_fl_iteration_num(uint64_t iteration_num) { iteration_num_ = iteration_num; }

uint64_t CloudWorker::fl_iteration_num() const { return iteration_num_.load(); }

void CloudWorker::set_data_size(int data_size) { data_size_ = data_size; }

void CloudWorker::set_secret_pk(armour::PrivateKey *secret_pk) { secret_pk_ = secret_pk; }

void CloudWorker::set_pw_salt(const std::vector<uint8_t> pw_salt) { pw_salt_ = pw_salt; }

void CloudWorker::set_pw_iv(const std::vector<uint8_t> pw_iv) { pw_iv_ = pw_iv; }

void CloudWorker::set_public_keys_list(const std::vector<EncryptPublicKeys> public_keys_list) {
  public_keys_list_ = public_keys_list;
}

int CloudWorker::data_size() const { return data_size_; }

armour::PrivateKey *CloudWorker::secret_pk() const { return secret_pk_; }

std::vector<uint8_t> CloudWorker::pw_salt() const { return pw_salt_; }

std::vector<uint8_t> CloudWorker::pw_iv() const { return pw_iv_; }

std::vector<EncryptPublicKeys> CloudWorker::public_keys_list() const { return public_keys_list_; }

std::string CloudWorker::fl_name() const { return fl_name_; }

std::string CloudWorker::fl_id() const { return worker_node_->fl_id(); }
}  // namespace worker
}  // namespace fl
}  // namespace mindspore
