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

#ifndef MINDSPORE_CCSRC_FL_COMMUNICATOR_MESSAGE_HANDLER_H_
#define MINDSPORE_CCSRC_FL_COMMUNICATOR_MESSAGE_HANDLER_H_

#include <string>
#include "common/utils/log_adapter.h"

namespace mindspore {
namespace fl {
typedef void (*RefBufferRelCallback)(const void *data, size_t datalen, void *extra);
// MessageHandler class is used to handle requests from clients and send response from server.
// It's the base class of HttpMsgHandler and TcpMsgHandler.
class MessageHandler {
 public:
  MessageHandler() = default;
  virtual ~MessageHandler() = default;

  // Raw data of this message in bytes.
  virtual const void *data() const = 0;

  // Raw data size of this message.(Number of bytes)
  virtual size_t len() const = 0;

  // string message type of this message.
  virtual std::string message_type() const = 0;

  // string message id of this message.
  virtual std::string message_id() const = 0;

  bool HasSentResponse() { return has_sent_response_; }
  virtual bool SendResponse(const void *data, const size_t &len) = 0;
  virtual bool SendResponse(const void *data, const size_t &len, const std::string &message_id) = 0;
  virtual bool SendResponseInference(const void *data, const size_t &len, RefBufferRelCallback cb) {
    auto ret = SendResponse(data, len);
    if (cb) {
      cb(data, len, nullptr);
    }
    has_sent_response_ = true;
    return ret;
  }

 protected:
  bool has_sent_response_ = false;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMMUNICATOR_MESSAGE_HANDLER_H_
