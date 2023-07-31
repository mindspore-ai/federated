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

#include "common/communicator/tcp_communicator.h"
#include <memory>
#include <utility>
#include "communicator/tcp_msg_handler.h"

namespace mindspore {
namespace fl {
namespace {
constexpr int kDefaultTcpThreadCount = 3;
const std::unordered_map<TcpUserCommand, std::string> kUserCommandToMsgType = {
  {TcpUserCommand::kPullWeight, "pullWeight"},  {TcpUserCommand::kPushWeight, "pushWeight"},
  {TcpUserCommand::kStartFLJob, "startFLJob"},  {TcpUserCommand::kExchangeKeys, "exchangeKeys"},
  {TcpUserCommand::kGetKeys, "getKeys"},        {TcpUserCommand::kUpdateModel, "updateModel"},
  {TcpUserCommand::kGetResult, "getResult"},    {TcpUserCommand::kGetModel, "getModel"},
  {TcpUserCommand::kPushMetrics, "pushMetrics"}};
}  // namespace

bool TcpCommunicator::Start() {
  if (task_executor_) {
    MS_LOG(INFO) << "The TCP communicator has already started.";
    return true;
  }
  task_executor_ = std::make_shared<TaskExecutor>(kDefaultTcpThreadCount);
  MS_EXCEPTION_IF_NULL(task_executor_);
  return true;
}

bool TcpCommunicator::Stop() {
  // stop handle tcp message
  if (task_executor_) {
    task_executor_->Stop();
  }
  return true;
}

void TcpCommunicator::RegisterRoundMsgCallback(const std::string &msg_type, const MessageCallback &cb) {
  MS_LOG(INFO) << "msg_type is: " << msg_type;
  msg_callbacks_.try_emplace(msg_type, cb);
}

void TcpCommunicator::HandleRoundRequest(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                         const Protos &type, const VectorPtr &data) {
  MS_EXCEPTION_IF_NULL(conn);
  auto ret = HandleRoundRequestInner(conn, meta, type, data);
  if (!ret.IsSuccess()) {
    MS_LOG_ERROR << ret.StatusMessage();
    conn->ErrorResponse(meta, ret.StatusMessage());
  }
}

FlStatus TcpCommunicator::HandleRoundRequestInner(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                                  const Protos &, const VectorPtr &data) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(data);
  TcpUserCommand user_command = static_cast<TcpUserCommand>(meta.user_cmd());
  auto msg_it = kUserCommandToMsgType.find(user_command);
  if (msg_it == kUserCommandToMsgType.end()) {
    std::stringstream stringstream;
    stringstream << "Tcp server doesn't support command " << user_command;
    return FlStatus(kFlFailed, stringstream.str());
  }
  const std::string &msg_type = msg_it->second;
  auto msg_handler = GetRoundMsgCallBack(msg_type);
  if (msg_handler == nullptr) {
    std::stringstream stringstream;
    stringstream << "Tcp server doesn't support command " << user_command << " " << msg_type;
    return FlStatus(kFlFailed, stringstream.str());
  }
  MS_LOG(DEBUG) << "TcpCommunicator receives message for " << msg_type;
  std::shared_ptr<MessageHandler> tcp_msg_handler = std::make_shared<TcpMsgHandler>(conn, meta, data);
  if (tcp_msg_handler == nullptr) {
    return FlStatus(kFlFailed, "Create TcpMsgHandler failed");
  }
  if (task_executor_ == nullptr) {
    return FlStatus(kFlFailed, "Task executor is not inited");
  }
  auto task = [msg_handler, tcp_msg_handler]() { msg_handler(tcp_msg_handler); };
  if (!task_executor_->Submit(task)) {
    return FlStatus(kFlFailed, "Submit tcp msg handler failed.");
  }
  return kFlSuccess;
}

CommunicatorBase::MessageCallback TcpCommunicator::GetRoundMsgCallBack(const std::string &msg_type) const {
  auto it = msg_callbacks_.find(msg_type);
  if (it == msg_callbacks_.end()) {
    return nullptr;
  }
  return it->second;
}
}  // namespace fl
}  // namespace mindspore
