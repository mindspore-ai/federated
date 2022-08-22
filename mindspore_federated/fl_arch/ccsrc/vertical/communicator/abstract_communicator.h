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



namespace mindspore {
namespace fl {
class AbstractCommunicator : public AbstractNode {
 public:
  AbstractCommunicator() = default;
  ~AbstractCommunicator() = default;
  virtual bool LaunchMsgHandler(const std::shared_ptr<MessageHandler> &message) = 0;
  void RegisterMsgCallBack(const std::shared_ptr<CommunicatorBase> &http_communicator, const std::string &name);
  std::shared_ptr<CommunicatorBase> CreateHttpCommunicator();
  void SendResponseMsg(const std::shared_ptr<MessageHandler> &message, const void *data, size_t len);
  bool verifyResponse(const std::shared_ptr<MessageHandler> &message, const void *data, size_t len);

 private:
  std::shared_ptr<HttpCommunicator> http_communicator_ = nullptr;
  std::mutex communicator_mutex_;
  std::shared_ptr<HttpServer> http_server_ = nullptr;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_ABSTRACT_COMMUNICATOR_H_
