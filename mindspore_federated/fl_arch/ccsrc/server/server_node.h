/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_NODE_H_
#define MINDSPORE_CCSRC_FL_SERVER_NODE_H_

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_map>
#include <map>
#include <functional>

#include "common/fl_context.h"
#include "communicator/tcp_client.h"
#include "communicator/tcp_server.h"
#include "common/core/abstract_node.h"
#include "communicator/communicator_base.h"

namespace mindspore {
namespace fl {
namespace server {
class ServerNode : public AbstractNode {
 public:
  ServerNode() = default;

  ~ServerNode() override = default;

  void InitializeBeforeCache(const std::string &ip, uint16_t port);

  bool Start(const uint32_t &timeout = FLContext::instance()->cluster_config().cluster_available_timeout) override;
  bool Stop() override;

  void BroadcastEvent(ServerBroadcastMessage broadcast_msg);
  bool ServerPingPong();
  bool GetModelWeight(uint64_t iteration_num, VectorPtr *output);
  void BroadcastModelWeight(const std::string &proto_model,
                            const std::map<std::string, std::string> &broadcast_server_map);
  bool PullWeight(const uint8_t *req_data, size_t len, VectorPtr *output);

 private:
  void Initialize();
  bool TcpMessageHandleSubclass(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                const Protos &protos, const VectorPtr &size) override;

  void HandleBroadcastEvent(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const Protos &protos,
                            const VectorPtr &size);
  void HandleServerPing(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const Protos &protos,
                        const VectorPtr &size);
  void HandleServerPong(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const Protos &protos,
                        const VectorPtr &size);
  void HandleGetModelWeight(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const Protos &protos,
                            const VectorPtr &size);
  void HandleBroadcastModelWeight(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                  const Protos &protos, const VectorPtr &data);
  void HandleServerPullWeight(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const Protos &protos,
                              const VectorPtr &data);
  void PingOneServer(const std::string &node_id, const std::string &tcp_address);
  void PongOneServer(const std::string &node_id, const std::string &tcp_address);
  std::vector<std::string> pong_received_servers_;
  std::map<std::string, std::string> try_visited_servers_;
  std::mutex ping_pong_mutex_;
  std::condition_variable ping_pong_cond_var_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_NODE_H_
