/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_ABSTRACT_NODE_H_
#define MINDSPORE_CCSRC_FL_ABSTRACT_NODE_H_

#include <utility>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include <unordered_map>
#include <functional>
#include <mutex>

#include "communicator/message.h"
#include "common/constants.h"
#include "common/core/node_info.h"
#include "common/communicator/task_executor.h"
#include "common/communicator/communicator_base.h"
#include "common/communicator/tcp_client.h"
#include "common/communicator/tcp_server.h"
#include "common/communicator/http_server.h"
#include "common/communicator/http_communicator.h"
#include "common/communicator/tcp_communicator.h"
#include "common/status.h"

namespace mindspore {
namespace fl {
constexpr int kCommTimeoutInSeconds = 10;

class AbstractNode;
class ResponseTrack {
 public:
  using MessageCallback = std::function<void(const MessageMeta &meta, const VectorPtr &response_data)>;

  ResponseTrack(AbstractNode *node, uint64_t request_id, uint64_t expect_count, const MessageCallback &callback);
  ~ResponseTrack();
  uint64_t request_id() const { return request_id_; }
  uint64_t expect_count() const { return expect_count_; }

  bool OnRecvResponseData();
  bool OnRecvResponseData(const MessageMeta &meta, const Protos &protos, const VectorPtr &data);
  bool CheckMessageTrack() const;

 private:
  AbstractNode *node_ = nullptr;
  uint64_t request_id_ = 0;
  uint64_t expect_count_ = 0;
  std::atomic_uint64_t curr_count_ = 0;
  MessageCallback callback_ = nullptr;
};

class AbstractNode {
 public:
  AbstractNode() = default;
  virtual ~AbstractNode() = default;

  virtual bool Start(const uint32_t &timeout) = 0;
  virtual bool Stop() = 0;

  std::string node_id() const;
  std::string tcp_address() const;
  void StartTcpServer(const std::string &ip, uint16_t port);
  std::shared_ptr<CommunicatorBase> GetOrCreateTcpComm();
  std::shared_ptr<CommunicatorBase> GetOrCreateHttpComm(const std::string &ip, uint16_t port);
  void StartHttpServer();

  // for collective data send(by tcp client) and recv(by tcp server)
  std::shared_ptr<ResponseTrack> CollectiveSendAsync(const std::string &recv_address,
                                                     const CollectiveMessageMeta &collective_meta, const void *data,
                                                     size_t size);
  bool CollectiveRecvWait(const CollectiveMessageMeta &expect_meta, size_t expect_size, VectorPtr *output,
                          const uint32_t &timeout = kCommTimeoutInSeconds);

  // for tcp server
  bool TcpMessageHandle(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const Protos &protos,
                        const VectorPtr &data);

  // for tcp and http client
  bool Wait(const std::shared_ptr<ResponseTrack> &request_track, const uint32_t &timeout = kCommTimeoutInSeconds);
  void ReleaseResponseTrack(uint64_t request_id);

  // when initializing the node, should initializing the node info.
  void InitNodeInfo(const NodeRole &role);
  void OnIterationUpdate();

 protected:
  // for collective data recv
  void HandleCollectiveData(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const Protos &,
                            const VectorPtr &data);
  bool CollectiveRecvWaitInner(const CollectiveMessageMeta &expect_meta, VectorPtr *output, const uint32_t &timeout);

  // for tcp server
  void NotifyMessageArrival(const MessageMeta &meta, const Protos &protos, const VectorPtr &data);
  void NotifyMessageArrival(const std::shared_ptr<ResponseTrack> &response_track);
  void ProcessRoundRequest(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const Protos &,
                           const VectorPtr &data);

  // for tcp client
  std::shared_ptr<TcpClient> GetOrCreateTcpClient(const std::string &server_address);
  std::shared_ptr<ResponseTrack> AddMessageTrack(const uint32_t &expected_response,
                                                 const ResponseTrack::MessageCallback &callback);

  virtual bool TcpMessageHandleSubclass(const std::shared_ptr<TcpConnection> &, const MessageMeta &, const Protos &,
                                        const VectorPtr &) {
    return false;
  }
  void StartTcpCommunicator();
  void StopTcpServer();
  void StopHttpServer();

  std::mutex client_mutex_;
  std::unordered_map<std::string, std::shared_ptr<TcpClient>> tcp_client_map_;

  // recv data of collective request: send node id, recv CollectiveMessageMeta and data
  std::unordered_map<std::string, std::vector<std::pair<CollectiveMessageMeta, VectorPtr>>> collective_received_data_;
  std::mutex collective_received_mutex_;
  std::condition_variable collective_received_cond_;

  // track response of tcp and http client request
  std::unordered_map<uint64_t, std::weak_ptr<ResponseTrack>> message_tracker_;
  std::mutex message_tracker_mutex_;
  std::condition_variable message_tracker_cond_;
  std::atomic_uint64_t next_request_id_ = 0;

  NodeInfo node_info_;

  std::shared_ptr<HttpServer> http_server_ = nullptr;
  std::shared_ptr<TcpServer> tcp_server_ = nullptr;
  std::shared_ptr<TcpClient> tcp_client_local_ = nullptr;
  std::shared_ptr<HttpCommunicator> http_communicator_ = nullptr;
  std::shared_ptr<TcpCommunicator> tcp_communicator_ = nullptr;
  std::mutex communicator_mutex_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_ABSTRACT_NODE_H_
