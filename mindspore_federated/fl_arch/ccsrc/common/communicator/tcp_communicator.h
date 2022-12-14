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

#ifndef MINDSPORE_CCSRC_FL_TCP_COMMUNICATOR_H_
#define MINDSPORE_CCSRC_FL_TCP_COMMUNICATOR_H_

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "common/communicator/task_executor.h"
#include "common/communicator/communicator_base.h"
#include "common/constants.h"
#include "common/status.h"
#include "common/communicator/tcp_server.h"

namespace mindspore {
namespace fl {
class TcpCommunicator : public CommunicatorBase {
 public:
  TcpCommunicator() = default;
  ~TcpCommunicator() = default;

  bool Start() override;
  bool Stop() override;

  void RegisterRoundMsgCallback(const std::string &msg_type, const MessageCallback &cb) override;
  void HandleRoundRequest(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const Protos &type,
                          const VectorPtr &data);

 private:
  std::unordered_map<std::string, MessageCallback> msg_callbacks_;

  // The task executor of the communicators. This helps server to handle network message concurrently. The tasks
  // submitted to this task executor is asynchronous.
  std::shared_ptr<TaskExecutor> task_executor_ = nullptr;
  MessageCallback GetRoundMsgCallBack(const std::string &msg_type) const;

  FlStatus HandleRoundRequestInner(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const Protos &,
                                   const VectorPtr &data);
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_TCP_COMMUNICATOR_H_
