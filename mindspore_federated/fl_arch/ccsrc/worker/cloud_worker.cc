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

  http_server_address_ = FLContext::instance()->http_server_address();
  if (FLContext::instance()->enable_ssl()) {
    http_server_address_ = "https://" + http_server_address_;
  } else {
    http_server_address_ = "http://" + http_server_address_;
  }

  MS_LOG(INFO) << "fl id is:" << fl_id() << ". Request will be sent to server domain:" << http_server_address_;
  http_client_ = std::make_shared<HttpClient>(http_server_address_);

  http_client_->SetMessageCallback([&](const std::shared_ptr<ResponseTrack> &response_track,
                                       const std::string &kernel_path,
                                       const std::shared_ptr<std::vector<unsigned char>> &response_msg) {
    if (handlers_.count(kernel_path) <= 0) {
      MS_LOG(WARNING) << "The kernel path of response message is invalid.";
      return;
    }
    MS_LOG(DEBUG) << "Received the response"
                  << ", kernel_path is " << kernel_path << ", request_id is " << response_track->request_id()
                  << ", response_msg size is " << response_msg->size();
    const auto &callback = handlers_[kernel_path];
    callback(response_msg);
    NotifyMessageArrival(response_track);
  });

  http_client_->Init();
}

bool CloudWorker::SendToServerSync(const std::string kernel_path, const std::string content_type, const void *data,
                                   size_t data_size) {
  MS_ERROR_IF_NULL_W_RET_VAL(data, false);
  if (data_size == 0) {
    MS_LOG(WARNING) << "Sending request for data size must be > 0";
    return false;
  }
  auto request_track = AddMessageTrack(1, nullptr);
  if (!http_client_->SendMessage(kernel_path, content_type, data, data_size, request_track)) {
    MS_LOG(WARNING) << "Sending request for " << kernel_path << " to server " << http_server_address_ << " failed.";
    return false;
  }
  if (!Wait(request_track)) {
    http_client_->BreakLoopEvent();
    return false;
  }
  return true;
}

void CloudWorker::RegisterMessageCallback(const std::string kernel_path, const MessageReceive &cb) {
  if (handlers_.count(kernel_path) > 0) {
    MS_LOG(DEBUG) << "Http handlers has already register kernel path:" << kernel_path;
    return;
  }
  handlers_[kernel_path] = cb;
  MS_LOG(INFO) << "Http handlers register kernel path:" << kernel_path;
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

std::string CloudWorker::fl_name() const { return kServerModeFL; }

std::string CloudWorker::fl_id() const { return worker_node_->fl_id(); }
}  // namespace worker
}  // namespace fl
}  // namespace mindspore
