/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "server/kernel/round/round_kernel.h"
#include <mutex>
#include <queue>
#include <chrono>
#include <thread>
#include <utility>
#include <string>
#include <vector>
#include "server/iteration.h"
#include "server/cert_verify.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
RoundKernel::RoundKernel() = default;

RoundKernel::~RoundKernel() = default;

void RoundKernel::InitKernelCommon(size_t iteration_time_window) {
  iteration_time_window_ = iteration_time_window;
  executor_ = &Executor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor_);
  if (!executor_->initialized()) {
    MS_LOG(EXCEPTION) << "Executor must be initialized in server pipeline.";
    return;
  }
}

void RoundKernel::OnFirstCountEvent() {}

void RoundKernel::OnLastCountEvent() {}

void RoundKernel::set_name(const std::string &name) { name_ = name; }

void RoundKernel::SendResponseMsg(const std::shared_ptr<MessageHandler> &message, const void *data, size_t len) {
  if (!verifyResponse(message, data, len)) {
    return;
  }
  IncreaseTotalClientNum();
  if (!message->SendResponse(data, len)) {
    MS_LOG(WARNING) << "Sending response failed.";
    return;
  }
  if (DataRateKernels.find(name_) != DataRateKernels.end()) {
    CalculateSendData(len);
  }
}

void RoundKernel::SendResponseMsgInference(const std::shared_ptr<MessageHandler> &message, const void *data, size_t len,
                                           RefBufferRelCallback cb) {
  if (!verifyResponse(message, data, len)) {
    return;
  }
  IncreaseTotalClientNum();
  if (!message->SendResponseInference(data, len, cb)) {
    MS_LOG(WARNING) << "Sending response failed.";
    return;
  }
  if (DataRateKernels.find(name_) != DataRateKernels.end()) {
    CalculateSendData(len);
  }
}

bool RoundKernel::verifyResponse(const std::shared_ptr<MessageHandler> &message, const void *data, size_t len) {
  if (message == nullptr) {
    MS_LOG(WARNING) << "The message handler is nullptr.";
    return false;
  }
  if (data == nullptr || len == 0) {
    std::string reason = "The output of the round " + name_ + " is empty.";
    MS_LOG(WARNING) << reason;
    if (!message->SendResponse(reason.c_str(), reason.size())) {
      MS_LOG(WARNING) << "Sending response failed.";
    }
    return false;
  }
  return true;
}

sigVerifyResult RoundKernel::VerifySignatureBase(const std::string &fl_id,
                                                 const std::vector<std::string> &src_data_list,
                                                 const flatbuffers::Vector<uint8_t> *signature,
                                                 const std::string &timestamp) {
  std::vector<unsigned char> src_data;
  for (auto &item : src_data_list) {
    src_data.insert(src_data.end(), item.begin(), item.end());
  }
  return VerifySignatureBase(fl_id, src_data, signature, timestamp);
}

sigVerifyResult RoundKernel::VerifySignatureBase(const std::string &fl_id, const std::vector<uint8_t> &src_data,
                                                 const flatbuffers::Vector<uint8_t> *signature,
                                                 const std::string &timestamp) {
  if (signature == nullptr) {
    MS_LOG(DEBUG) << "signature in request " << name_ << " is nullptr";
    return sigVerifyResult::FAILED;
  }
  std::string key_attestation;
  auto found = cache::ClientInfos::GetInstance().GetClientKeyAttestation(fl_id, &key_attestation);
  if (!found.IsSuccess()) {
    MS_LOG(WARNING) << "can not find key attestation for fl_id: " << fl_id;
    return sigVerifyResult::TIMEOUT;
  }
  auto &certVerify = CertVerify::GetInstance();
  unsigned char srcDataHash[SHA256_DIGEST_LENGTH];
  certVerify.sha256Hash(src_data.data(), SizeToInt(src_data.size()), srcDataHash, SHA256_DIGEST_LENGTH);
  if (!certVerify.verifyRSAKey(key_attestation, srcDataHash, signature->data(), SHA256_DIGEST_LENGTH)) {
    return sigVerifyResult::FAILED;
  }
  if (!certVerify.verifyTimeStamp(fl_id, timestamp)) {
    return sigVerifyResult::TIMEOUT;
  }
  MS_LOG(INFO) << "verify signature for fl_id: " << fl_id << " success.";
  return sigVerifyResult::PASSED;
}

void RoundKernel::IncreaseTotalClientNum() { total_client_num_ += 1; }

void RoundKernel::IncreaseAcceptClientNum() { accept_client_num_ += 1; }

void RoundKernel::Summarize() {
  if (name_ == "startFLJob" || name_ == "updateModel" || name_ == "getModel") {
    MS_LOG(INFO) << "Round kernel " << name_ << " total client num is: " << total_client_num_
                 << ", accept client num is: " << accept_client_num_
                 << ", reject client num is: " << (total_client_num_ - accept_client_num_);
  }

  if (name_ == "updateModel" && accept_client_num() > 0) {
    MS_LOG(INFO) << "Client upload avg loss: " << (upload_loss_ / train_data_size_)
                 << ", upload avg accuracy: " << (upload_accuracy_ / eval_data_size_);
  }
}

size_t RoundKernel::total_client_num() const { return total_client_num_; }

size_t RoundKernel::accept_client_num() const { return accept_client_num_; }

size_t RoundKernel::reject_client_num() const { return total_client_num_ - accept_client_num_; }

void RoundKernel::InitClientVisitedNum() {
  total_client_num_ = 0;
  accept_client_num_ = 0;
}

void RoundKernel::InitClientUploadLoss() { upload_loss_ = 0.0f; }

void RoundKernel::InitClientUploadAccuracy() { upload_accuracy_ = 0.0f; }

void RoundKernel::InitEvalDataSize() { eval_data_size_ = 0; }

void RoundKernel::InitTrainDataSize() { train_data_size_ = 0; }

void RoundKernel::UpdateClientUploadLoss(const float upload_loss, const float data_size) {
  upload_loss_ = upload_loss_ + upload_loss * data_size;
  train_data_size_ += data_size;
}

void RoundKernel::UpdateClientUploadAccuracy(const float upload_accuracy, const size_t eval_data_size) {
  upload_accuracy_ = upload_accuracy_ + upload_accuracy * eval_data_size;
  eval_data_size_ += eval_data_size;
}

float RoundKernel::upload_loss() const { return upload_loss_; }

size_t RoundKernel::eval_data_size() const { return eval_data_size_; }

size_t RoundKernel::train_data_size() const { return train_data_size_; }

float RoundKernel::upload_accuracy() const { return upload_accuracy_; }

void RoundKernel::CalculateSendData(size_t send_len) {
  uint64_t second_time_stamp =
    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  if (send_data_time_ == 0) {
    send_data_ = send_len;
    send_data_time_ = second_time_stamp;
    return;
  }
  if (second_time_stamp == send_data_time_) {
    send_data_ += send_len;
  } else {
    RecordSendData(send_data_time_, send_data_);
    send_data_time_ = second_time_stamp;
    send_data_ = send_len;
  }
}

void RoundKernel::CalculateReceiveData(size_t receive_len) {
  uint64_t second_time_stamp =
    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  if (receive_data_time_ == 0) {
    receive_data_time_ = second_time_stamp;
    receive_data_ = receive_len;
    return;
  }
  if (second_time_stamp == receive_data_time_) {
    receive_data_ += receive_len;
  } else {
    RecordReceiveData(receive_data_time_, receive_data_);
    receive_data_time_ = second_time_stamp;
    receive_data_ = receive_len;
  }
}

void RoundKernel::RecordSendData(uint64_t time_stamp_second, size_t send_data) {
  std::lock_guard<std::mutex> lock(send_data_rate_mutex_);
  send_data_and_time_[time_stamp_second] = send_data;
}

void RoundKernel::RecordReceiveData(uint64_t time_stamp_second, size_t receive_data) {
  std::lock_guard<std::mutex> lock(receive_data_rate_mutex_);
  receive_data_and_time_[time_stamp_second] = receive_data;
}

std::map<uint64_t, size_t> RoundKernel::GetSendData() {
  std::lock_guard<std::mutex> lock(send_data_rate_mutex_);
  return send_data_and_time_;
}

std::map<uint64_t, size_t> RoundKernel::GetReceiveData() {
  std::lock_guard<std::mutex> lock(receive_data_rate_mutex_);
  return receive_data_and_time_;
}

void RoundKernel::ClearData() {
  std::lock_guard<std::mutex> lock(send_data_rate_mutex_);
  std::lock_guard<std::mutex> lock2(receive_data_rate_mutex_);
  send_data_and_time_.clear();
  receive_data_and_time_.clear();
}
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
