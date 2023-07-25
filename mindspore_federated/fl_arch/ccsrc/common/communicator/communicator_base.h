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

#ifndef MINDSPORE_CCSRC_FL_COMMUNICATOR_COMMUNICATOR_BASE_H_
#define MINDSPORE_CCSRC_FL_COMMUNICATOR_COMMUNICATOR_BASE_H_

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <thread>

#include "common/protos/comm.pb.h"
#include "communicator/message_handler.h"
#include "common/utils/log_adapter.h"

namespace mindspore {
namespace fl {
enum class TcpUserCommand {
  kPullWeight,
  kPushWeight,
  kStartFLJob,
  kUpdateModel,
  kGetModel,
  kPushMetrics,
  kExchangeKeys,
  kGetKeys,
  kGetResult,
};

using VectorPtr = std::shared_ptr<std::vector<uint8_t>>;

// CommunicatorBase is used to receive request and send response for server.
// It is the base class of HttpCommunicator and TcpCommunicator.
class CommunicatorBase {
 public:
  CommunicatorBase() = default;
  virtual ~CommunicatorBase() = default;

  virtual bool Start() = 0;
  virtual bool Stop() = 0;

  using MessageCallback = std::function<void(const std::shared_ptr<MessageHandler> &)>;
  virtual void RegisterRoundMsgCallback(const std::string &msg_type, const MessageCallback &cb) = 0;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMMUNICATOR_COMMUNICATOR_BASE_H_
