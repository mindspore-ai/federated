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

#ifndef MINDSPORE_CCSRC_FL_COMMUNICATOR_HTTP_MSG_HANDLER_H_
#define MINDSPORE_CCSRC_FL_COMMUNICATOR_HTTP_MSG_HANDLER_H_

#include <memory>
#include <string>
#include "common/communicator/http_message_handler.h"
#include "common/communicator/message_handler.h"

namespace mindspore {
namespace fl {
constexpr int kHttpSuccess = 200;
class HttpMsgHandler : public MessageHandler {
 public:
  HttpMsgHandler(const std::shared_ptr<HttpMessageHandler> &http_msg, void *data, size_t len, std::string message_type,
                 std::string message_id, std::string message_source, std::string message_offset);
  ~HttpMsgHandler() override = default;

  const void *data() const override;
  size_t len() const override;
  std::string message_type() const override;
  std::string message_id() const override;
  std::string message_source() const override;
  std::string message_offset() const override;

  bool SendResponse(const void *data, const size_t &len) override;
  bool SendResponse(const void *data, const size_t &len, const std::string &message_id) override;
  bool SendResponseInference(const void *data, const size_t &len, RefBufferRelCallback cb) override;

 private:
  std::shared_ptr<HttpMessageHandler> http_msg_;
  void *data_;
  size_t len_;
  // We use message type to make sure the message pattern
  std::string message_type_;
  // We use message id to track message request and response
  std::string message_id_;
  // We use message source to make sure which server sends the message
  std::string message_source_;
  // We use message offset to realize the slices of the message
  std::string message_offset_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMMUNICATOR_HTTP_MSG_HANDLER_H_
