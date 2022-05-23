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

#ifndef MINDSPORE_CCSRC_FL_COMMUNICATOR_HTTP_COMMUNICATOR_H_
#define MINDSPORE_CCSRC_FL_COMMUNICATOR_HTTP_COMMUNICATOR_H_

#include <string>
#include <memory>
#include <unordered_map>
#include "common/communicator/http_server.h"
#include "common/communicator/http_message_handler.h"
#include "common/communicator/communicator_base.h"
#include "common/communicator/http_msg_handler.h"

namespace mindspore {
namespace fl {
class HttpCommunicator : public CommunicatorBase {
 public:
  explicit HttpCommunicator(const std::shared_ptr<HttpServer> &http_server) : http_server_(http_server) {}
  ~HttpCommunicator() = default;

  bool Start() override { return true; }
  bool Stop() override { return true; }
  void RegisterRoundMsgCallback(const std::string &msg_type, const MessageCallback &cb) override;

 private:
  std::shared_ptr<HttpServer> http_server_;
  using HttpMsgCallback = std::function<void(const std::shared_ptr<HttpMessageHandler> &)>;
  std::unordered_map<std::string, HttpMsgCallback> http_msg_callbacks_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMMUNICATOR_HTTP_COMMUNICATOR_H_
