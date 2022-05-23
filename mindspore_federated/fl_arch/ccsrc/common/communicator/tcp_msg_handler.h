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

#ifndef MINDSPORE_CCSRC_FL_TCP_MSG_HANDLER_H_
#define MINDSPORE_CCSRC_FL_TCP_MSG_HANDLER_H_

#include <memory>
#include "communicator/tcp_communicator.h"
#include "communicator/message_handler.h"

namespace mindspore {
namespace fl {
class TcpMsgHandler : public MessageHandler {
 public:
  TcpMsgHandler(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const VectorPtr &data);
  ~TcpMsgHandler() override = default;

  const void *data() const override;
  size_t len() const override;
  bool SendResponse(const void *data, const size_t &len) override;

 private:
  std::shared_ptr<TcpConnection> tcp_conn_;
  // MessageMeta is used for server to get the user command and to find communication peer when responding.
  MessageMeta meta_;
  // We use data of shared_ptr array so that the raw pointer won't be released until the reference is 0.
  VectorPtr data_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_TCP_MSG_HANDLER_H_
