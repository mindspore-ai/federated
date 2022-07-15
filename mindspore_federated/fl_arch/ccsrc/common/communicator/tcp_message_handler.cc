/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "common/communicator/tcp_message_handler.h"

namespace mindspore {
namespace fl {
void TcpMessageHandler::ReceiveMessage(const ReadBufferFun &read_fun) {
  while (true) {
    bool end_read = false;
    try {
      auto ret = ReceiveMessageInner(read_fun, &end_read);
      if (!ret) {
        Reset();
      }
    } catch (const std::bad_alloc &) {
      MS_LOG_WARNING << "Catch exception std::bad_alloc";
      Reset();
    }
    if (end_read) {
      return;
    }
  }
}

bool TcpMessageHandler::ReadMessageHeader(const ReadBufferFun &read_fun, bool *end_read) {
  if (cur_header_len_ >= kHeaderLen) {
    return true;
  }
  size_t expect_size = kHeaderLen - cur_header_len_;
  auto read_size = read_fun(header_ + cur_header_len_, expect_size);
  if (read_size >= 0) {
    cur_header_len_ += read_size;
  }
  if (read_size < expect_size) {
    *end_read = true;
    return true;
  }
  if (cur_header_len_ == kHeaderLen) {
    message_header_ = *(reinterpret_cast<MessageHeader *>(header_));
    if (message_header_.message_proto_ != Protos::RAW && message_header_.message_proto_ != Protos::FLATBUFFERS &&
        message_header_.message_proto_ != Protos::PROTOBUF) {
      MS_LOG(WARNING) << "The proto:" << message_header_.message_proto_ << " is illegal!";
      return false;
    }
    if (message_header_.message_length_ == 0 || message_header_.message_meta_length_ == 0) {
      MS_LOG_WARNING << "The message len " << message_header_.message_length_ << " or meta length "
                     << message_header_.message_meta_length_ << " is invalid!";
      return false;
    }
    if (message_header_.message_length_ >= INT32_MAX) {
      MS_LOG(WARNING) << "The message len:" << message_header_.message_length_ << " is too long.";
      return false;
    }
    if (message_header_.message_meta_length_ >= message_header_.message_length_) {
      MS_LOG(WARNING) << "The message meta len " << message_header_.message_meta_length_ << " >= the message len "
                      << message_header_.message_length_;
      return false;
    }
    meta_buffer_.resize(message_header_.message_meta_length_);
    auto message_data_len = message_header_.message_length_ - message_header_.message_meta_length_;
    data_ = std::make_shared<std::vector<uint8_t>>();
    if (data_ == nullptr) {
      MS_LOG(WARNING) << "New message data shared_ptr failed";
      return false;
    }
    data_->resize(message_data_len);
  }
  return true;
}

bool TcpMessageHandler::ReadMessageMeta(const ReadBufferFun &read_fun, bool *end_read) {
  if (cur_meta_len_ >= meta_buffer_.size()) {
    return true;
  }
  size_t expect_size = meta_buffer_.size() - cur_meta_len_;
  auto read_size = read_fun(meta_buffer_.data() + cur_meta_len_, expect_size);
  if (read_size >= 0) {
    cur_meta_len_ += read_size;
  }
  if (read_size < expect_size) {
    *end_read = true;
    return true;
  }
  if (cur_meta_len_ == meta_buffer_.size()) {
    if (!message_meta_.ParseFromArray(meta_buffer_.data(), static_cast<int>(meta_buffer_.size()))) {
      MS_LOG(WARNING) << "Parse protobuf MessageMeta failed";
      return false;
    }
  }
  return true;
}

bool TcpMessageHandler::ReadMessageDataAndCallback(const ReadBufferFun &read_fun, bool *end_read) {
  if (data_ == nullptr) {
    MS_LOG_WARNING << "Data cannot be nullptr";
    return false;
  }
  // data_->size() != 0
  if (cur_data_len_ >= data_->size()) {
    return true;
  }
  size_t expect_size = data_->size() - cur_data_len_;
  auto read_size = read_fun(data_->data() + cur_data_len_, expect_size);
  if (read_size >= 0) {
    cur_data_len_ += read_size;
  }
  if (read_size < expect_size) {
    *end_read = true;
    return true;
  }
  if (cur_data_len_ == data_->size()) {
    if (msg_callback_) {
      try {
        msg_callback_(message_meta_, message_header_.message_proto_, data_);
      } catch (const std::exception &e) {
        MS_LOG_WARNING << "Catch exception when handle tcp message: " << e.what()
                       << ", msg meta cmd: " << message_meta_.cmd();
      }
    }
    Reset();
  }
  return true;
}

bool TcpMessageHandler::ReceiveMessageInner(const ReadBufferFun &read_fun, bool *end_read) {
  if (read_fun == nullptr || end_read == nullptr) {
    return false;
  }
  if (!ReadMessageHeader(read_fun, end_read)) {
    return false;
  }
  if (*end_read) {
    return true;
  }
  if (!ReadMessageMeta(read_fun, end_read)) {
    return false;
  }
  if (*end_read) {
    return true;
  }
  return ReadMessageDataAndCallback(read_fun, end_read);
}

void TcpMessageHandler::Reset() {
  cur_header_len_ = 0;
  cur_meta_len_ = 0;
  cur_data_len_ = 0;
  meta_buffer_.clear();
  data_ = nullptr;
}
}  // namespace fl
}  // namespace mindspore
