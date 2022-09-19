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
#ifndef MINDSPORE_FL_ARCH_CCSRC_VERTICAL_ABSTRACT_COMMUNICATOR_H_
#define MINDSPORE_FL_ARCH_CCSRC_VERTICAL_ABSTRACT_COMMUNICATOR_H_

#include <string>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <map>
#include "vertical/vfl_context.h"
#include "common/communicator/http_communicator.h"
#include "common/communicator/http_server.h"
#include "common/utils/log_adapter.h"
#include "common/core/abstract_node.h"
#include "common/communicator/http_client.h"

namespace mindspore {
namespace fl {
class AbstractCommunicator : public AbstractNode {
 public:
  AbstractCommunicator() = default;
  ~AbstractCommunicator() = default;

  virtual bool LaunchMsgHandler(const std::shared_ptr<MessageHandler> &message) = 0;

  virtual void InitCommunicator(const std::shared_ptr<HttpCommunicator> &http_communicator) = 0;

  bool Start(const uint32_t &timeout = 1000) override { return true; }

  bool Stop() override { return true; }

  void RegisterMsgCallBack(const std::shared_ptr<HttpCommunicator> &http_communicator, const std::string &name);

  static std::shared_ptr<HttpCommunicator> CreateHttpCommunicator();

  static void StartHttpServer(const std::shared_ptr<HttpCommunicator> &http_communicator);

  void SendResponseMsg(const std::shared_ptr<MessageHandler> &message, const void *data, size_t len);

  bool verifyResponse(const std::shared_ptr<MessageHandler> &message, const void *data, size_t len);

  void InitHttpClient();

  bool SendMessage(const void *data, size_t data_size, const std::string &msg_type);

 private:
  std::string remote_server_address_;

  std::shared_ptr<HttpClient> http_client_ = nullptr;

  static std::mutex communicator_mtx_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_ABSTRACT_COMMUNICATOR_H_
