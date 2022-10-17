/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "common/communicator/http_msg_handler.h"
#include <memory>
#include <string>

namespace mindspore {
namespace fl {
HttpMsgHandler::HttpMsgHandler(const std::shared_ptr<HttpMessageHandler> &http_msg, void *data, size_t len,
                               std::string message_type, std::string message_id)
    : http_msg_(http_msg), data_(data), len_(len), message_type_(message_type), message_id_(message_id) {}

const void *HttpMsgHandler::data() const {
  MS_ERROR_IF_NULL_W_RET_VAL(data_, nullptr);
  return data_;
}

size_t HttpMsgHandler::len() const { return len_; }

std::string HttpMsgHandler::message_type() const { return message_type_; }

std::string HttpMsgHandler::message_id() const { return message_id_; }

bool HttpMsgHandler::SendResponse(const void *data, const size_t &len) {
  MS_ERROR_IF_NULL_W_RET_VAL(data, false);
  http_msg_->QuickResponse(kHttpSuccess, data, len);
  has_sent_response_ = true;
  return true;
}

bool HttpMsgHandler::SendResponse(const void *data, const size_t &len, const std::string &message_id) {
  MS_ERROR_IF_NULL_W_RET_VAL(data, false);
  http_msg_->AddRespHeadParam("Message-Id", message_id);
  http_msg_->QuickResponse(kHttpSuccess, data, len);
  has_sent_response_ = true;
  return true;
}

bool HttpMsgHandler::SendResponseInference(const void *data, const size_t &len, RefBufferRelCallback cb) {
  MS_ERROR_IF_NULL_W_RET_VAL(data, false);
  http_msg_->QuickResponseInference(kHttpSuccess, data, len, cb);
  has_sent_response_ = true;
  return true;
}
}  // namespace fl
}  // namespace mindspore
