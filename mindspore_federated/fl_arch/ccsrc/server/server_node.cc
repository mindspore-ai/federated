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
#include "server/server_node.h"
#include <map>
#include "distributed_cache/server.h"
#include "distributed_cache/counter.h"
#include "distributed_cache/instance_context.h"
#include "distributed_cache/iteration_task_thread.h"
#include "common/common.h"
#include "server/iteration.h"

namespace mindspore {
namespace fl {
namespace server {
void ServerNode::InitializeBeforeCache(const std::string &ip, uint16_t port) {
  StartTcpServer(ip, port);
  InitNodeInfo(NodeRole::SERVER);
  MS_LOG(INFO) << "[Server start]: Server node create tcp server successful!";
}

bool ServerNode::Start(const uint32_t &timeout) {
  MS_LOG(INFO) << "[Server start]: The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << " start communicator and http server!";
  StartTcpCommunicator();
  StartHttpServer();
  MS_LOG(INFO) << "[Server start]: Successfully start server node!";
  return true;
}

void ServerNode::Initialize() {}

bool ServerNode::Stop() {
  MS_LOG(INFO) << "Begin stop server node!";
  // Pause receiving client messages and events
  cache::InstanceContext::Instance().SetSafeMode(true);
  // all client request should be finished
  Iteration::GetInstance().WaitAllRoundsFinish();
  // Counter and timer handling including AllReduce should be finished
  cache::IterationTaskThread::Instance().WaitAllTaskFinish();
  StopTcpServer();
  StopHttpServer();
  MS_LOG(INFO) << "End stop server node!";
  return true;
}

bool ServerNode::TcpMessageHandleSubclass(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                          const Protos &protos, const VectorPtr &data) {
  switch (meta.cmd()) {
    case NodeCommand::SERVER_BROADCAST_EVENT:
      HandleBroadcastEvent(conn, meta, protos, data);
      break;
    case NodeCommand::SERVER_PING:
      HandleServerPing(conn, meta, protos, data);
      break;
    case NodeCommand::SERVER_PONG:
      HandleServerPong(conn, meta, protos, data);
      break;
    case NodeCommand::GET_MODEL_WEIGHT:
      HandleGetModelWeight(conn, meta, protos, data);
      break;
    case NodeCommand::BROADCAST_MODEL_WEIGHT:
      HandleBroadcastModelWeight(conn, meta, protos, data);
      break;
    case NodeCommand::SERVER_PULL_WEIGHT:
      HandleServerPullWeight(conn, meta, protos, data);
      break;
    default:
      return false;
  }
  return true;
}

void ServerNode::BroadcastEvent(ServerBroadcastMessage broadcast_msg) {
  MS_LOG_INFO << "Begin broadcast event " << static_cast<int>(broadcast_msg.type());
  auto node_map = cache::Server::Instance().GetAllServers();
  auto iteration_num = cache::InstanceContext::Instance().iteration_num();
  broadcast_msg.set_cur_iteration_num(iteration_num);
  const auto &send_node = node_info_.node_id_;
  auto const &broadcast_msg_str = broadcast_msg.SerializeAsString();
  for (auto &item : node_map) {
    const auto &recv_node = item.first;
    if (send_node == recv_node) {
      continue;
    }
    const auto &recv_address = item.second;
    auto tcp_client = GetOrCreateTcpClient(recv_address);
    if (tcp_client == nullptr) {
      MS_LOG_WARNING << "Failed to connect to server, node id: " << recv_node << ", node tcp address: " << recv_address;
      continue;
    }
    auto request_track = AddMessageTrack(1, nullptr);
    MessageMeta message_meta;
    message_meta.set_cmd(NodeCommand::SERVER_BROADCAST_EVENT);
    message_meta.set_request_id(request_track->request_id());
    message_meta.set_iteration_num(iteration_num);
    message_meta.set_send_node(send_node);
    message_meta.set_recv_node(recv_node);
    message_meta.set_role(node_info_.node_role_);
    tcp_client->SendMessage(message_meta, Protos::PROTOBUF, broadcast_msg_str.data(), broadcast_msg_str.size());
  }
  MS_LOG_INFO << "End broadcast event " << static_cast<int>(broadcast_msg.type());
}

void ServerNode::HandleBroadcastEvent(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                      const Protos &protos, const VectorPtr &data) {
  MS_LOG_INFO << "Receive broadcast event from " << meta.send_node();
  conn->SimpleResponse(meta);
  if (meta.recv_node() != node_id()) {
    MS_LOG_WARNING << "The recv node " << meta.recv_node() << " of server broadcast message != the node id "
                   << node_id() << " of current server";
    return;
  }
  ServerBroadcastMessage broadcast_msg;
  if (!broadcast_msg.ParseFromArray(data->data(), static_cast<int>(data->size()))) {
    MS_LOG_WARNING << "Failed to parse server broadcast message, data size: " << data->size()
                   << ", send node: " << meta.send_node() << ", current node: " << meta.recv_node();
    return;
  }
  auto iteration_num = cache::InstanceContext::Instance().iteration_num();
  if (broadcast_msg.cur_iteration_num() != iteration_num) {
    MS_LOG_INFO << "The iteration num " << broadcast_msg.cur_iteration_num()
                << " of server broadcast message != the iteration num " << iteration_num << " of current server";
    return;
  }
  switch (broadcast_msg.type()) {
    case ServerBroadcastMessage_BroadcastEventType_COUNT_EVENT:
      MS_LOG_INFO << "Receive count event from " << meta.send_node();
      cache::Counter::Instance().OnNotifyCountEvent(broadcast_msg);
      break;
    default:
      MS_LOG_WARNING << "Unexpected broadcast message " << static_cast<int>(broadcast_msg.type());
  }
}

bool ServerNode::ServerPingPong() {
  MS_LOG_INFO << "Begin ping all other servers";
  const auto &send_node = node_info_.node_id_;
  constexpr int ping_max_retry_times = 15;
  for (int i = 0; i < ping_max_retry_times; i++) {
    std::map<std::string, std::string> node_map;
    auto cache_ret = cache::Server::Instance().GetAllServersRealtime(&node_map);
    if (cache_ret.IsSuccess()) {
      std::unique_lock<std::mutex> lock(ping_pong_mutex_);
      try_visited_servers_.clear();
      for (auto &item : node_map) {
        const auto &recv_node = item.first;
        if (send_node == recv_node) {
          continue;
        }
        auto &servers = pong_received_servers_;
        if (std::find(servers.begin(), servers.end(), recv_node) == servers.end()) {
          try_visited_servers_[recv_node] = item.second;
        }
      }
      if (try_visited_servers_.empty()) {
        MS_LOG_INFO << "Success receive pong message from servers: " << pong_received_servers_;
        pong_received_servers_.clear();
        return true;
      }
      lock.unlock();
      for (auto &server : try_visited_servers_) {
        PingOneServer(server.first, server.second);
      }
    }
    std::unique_lock<std::mutex> lock(ping_pong_mutex_);
    ping_pong_cond_var_.wait_for(lock, std::chrono::seconds(1));
  }
  MS_LOG_ERROR << "Failed to receive pong messages from servers " << try_visited_servers_;
  std::unique_lock<std::mutex> lock(ping_pong_mutex_);
  pong_received_servers_.clear();
  try_visited_servers_.clear();
  return false;
}

void ServerNode::PingOneServer(const std::string &recv_node_id, const std::string &recv_tcp_address) {
  const auto &send_node = node_info_.node_id_;
  auto send_node_address = node_info_.ip_ + ":" + std::to_string(node_info_.port_);
  MS_LOG_INFO << "Send ping message to " << recv_node_id;
  auto tcp_client = GetOrCreateTcpClient(recv_tcp_address);
  if (tcp_client == nullptr) {
    MS_LOG_WARNING << "Failed to connect to server, node id: " << recv_node_id
                   << ", node tcp address: " << recv_tcp_address;
    return;
  }
  auto request_track = AddMessageTrack(1, nullptr);
  MessageMeta message_meta;
  message_meta.set_cmd(NodeCommand::SERVER_PING);
  message_meta.set_request_id(request_track->request_id());
  message_meta.set_send_node(send_node);
  message_meta.set_recv_node(recv_node_id);
  message_meta.set_role(node_info_.node_role_);
  auto ret = tcp_client->SendMessage(message_meta, Protos::RAW, send_node_address.data(), send_node_address.size());
  if (!ret) {
    MS_LOG(WARNING) << "Send ping message to tcp server " << recv_tcp_address << " failed";
  }
}

void ServerNode::PongOneServer(const std::string &recv_node_id, const std::string &recv_tcp_address) {
  const auto &send_node = node_info_.node_id_;
  auto send_node_address = node_info_.ip_ + ":" + std::to_string(node_info_.port_);
  MS_LOG_INFO << "Send pong message to " << recv_node_id;
  auto tcp_client = GetOrCreateTcpClient(recv_tcp_address);
  if (tcp_client == nullptr) {
    MS_LOG_WARNING << "Failed to connect to server, node id: " << recv_node_id
                   << ", node tcp address: " << recv_tcp_address;
    return;
  }
  auto request_track = AddMessageTrack(1, nullptr);
  MessageMeta message_meta;
  message_meta.set_cmd(NodeCommand::SERVER_PONG);
  message_meta.set_request_id(request_track->request_id());
  message_meta.set_send_node(send_node);
  message_meta.set_recv_node(recv_node_id);
  message_meta.set_role(node_info_.node_role_);
  auto ret = tcp_client->SendMessage(message_meta, Protos::RAW, send_node_address.data(), send_node_address.size());
  if (!ret) {
    MS_LOG(WARNING) << "Send ping message to tcp server " << recv_tcp_address << " failed";
  }
}

void ServerNode::HandleServerPing(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                  const Protos &protos, const VectorPtr &data) {
  conn->SimpleResponse(meta);
  const auto &send_node_id = meta.send_node();
  MS_LOG_INFO << "Receive ping message from " << send_node_id;
  auto data_uint8 = data->data();
  auto send_node_tcp_address = std::string(data_uint8, data_uint8 + data->size());
  PongOneServer(send_node_id, send_node_tcp_address);
}

void ServerNode::HandleServerPong(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                  const Protos &protos, const VectorPtr &data) {
  conn->SimpleResponse(meta);
  const auto &send_node_id = meta.send_node();
  MS_LOG_INFO << "Receive pong message from " << send_node_id;
  std::unique_lock<std::mutex> lock(ping_pong_mutex_);
  pong_received_servers_.push_back(send_node_id);
  bool all_received = true;
  for (auto &item : try_visited_servers_) {
    auto &node_id = item.first;
    auto &servers = pong_received_servers_;
    if (std::find(servers.begin(), servers.end(), node_id) == servers.end()) {
      all_received = false;
      break;
    }
  }
  lock.unlock();
  if (all_received) {
    ping_pong_cond_var_.notify_one();
  }
}

bool ServerNode::GetModelWeight(uint64_t iteration_num, VectorPtr *output) {
  MS_LOG_INFO << "Begin get model weight of iteration " << iteration_num << " from other servers";
  if (output == nullptr) {
    return false;
  }
  *output = nullptr;
  std::map<std::string, std::string> node_map;
  auto cache_ret = cache::Server::Instance().GetAllServersRealtime(&node_map);
  if (!cache_ret.IsSuccess()) {
    return false;
  }
  const auto &send_node = node_info_.node_id_;
  for (auto &item : node_map) {
    const auto &recv_node = item.first;
    if (send_node == recv_node) {
      continue;
    }
    const auto &recv_address = item.second;
    auto tcp_client = GetOrCreateTcpClient(recv_address);
    if (tcp_client == nullptr) {
      MS_LOG_WARNING << "Failed to connect to server, node id: " << recv_node << ", node tcp address: " << recv_address;
      continue;
    }
    auto request_track =
      AddMessageTrack(1, [output, recv_node](const MessageMeta &meta, const VectorPtr &response_data) {
        if (!meta.response_error().empty()) {
          return;
        }
        *output = response_data;
      });
    MessageMeta message_meta;
    message_meta.set_cmd(NodeCommand::GET_MODEL_WEIGHT);
    message_meta.set_request_id(request_track->request_id());
    message_meta.set_iteration_num(iteration_num);
    message_meta.set_send_node(send_node);
    message_meta.set_recv_node(recv_node);
    message_meta.set_role(node_info_.node_role_);
    uint64_t void_data = 0;
    auto ret = tcp_client->SendMessage(message_meta, Protos::RAW, &void_data, sizeof(void_data));
    if (!ret) {
      MS_LOG(WARNING) << "Get model weight from tcp server " << recv_address << " failed";
      continue;
    }
    constexpr int timeout_in_seconds_wait_response = 30;
    ret = Wait(request_track, timeout_in_seconds_wait_response);
    if (!ret) {
      continue;
    }
    if (*output == nullptr) {
      continue;
    }
    MS_LOG_INFO << "Success to get model weight of iteration " << iteration_num
                << " from other servers, model size: " << (*output)->size();
    return true;
  }
  MS_LOG_INFO << "Failed to get model weight of iteration " << iteration_num << " from other servers";
  return false;
}

void ServerNode::HandleGetModelWeight(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                      const Protos &protos, const VectorPtr &) {
  MS_LOG_INFO << "Begin handle get model weight message";
  ProtoModel proto_model;
  auto ret = Executor::GetInstance().GetModelByIteration(meta.iteration_num(), &proto_model);
  if (!ret) {
    auto error_msg = "Failed to get model of iteration " + std::to_string(meta.iteration_num());
    MS_LOG_INFO << error_msg;
    conn->ErrorResponse(meta, error_msg);
    return;
  }
  auto model_str = proto_model.SerializeAsString();
  if (!conn->SendMessage(meta, Protos::PROTOBUF, model_str.data(), model_str.size())) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
  MS_LOG_INFO << "End handle get model weight message";
}

void ServerNode::BroadcastModelWeight(const std::string &proto_model,
                                      const std::map<std::string, std::string> &broadcast_server_map) {
  MS_LOG_INFO << "Begin broadcast model weight";
  std::map<std::string, std::string> node_map;
  if (broadcast_server_map.empty()) {
    auto cache_ret = cache::Server::Instance().GetAllServersRealtime(&node_map);
    if (!cache_ret.IsSuccess()) {
      MS_LOG_WARNING << "Failed to get all servers real-time from cache";
      return;
    }
  } else {
    node_map = broadcast_server_map;
  }
  const auto &send_node = node_info_.node_id_;
  auto node_address = node_info_.ip_ + ":" + std::to_string(node_info_.port_);
  auto iteration_num = cache::InstanceContext::Instance().iteration_num();
  for (auto &item : node_map) {
    const auto &recv_node = item.first;
    if (send_node == recv_node) {
      continue;
    }
    const auto &recv_address = item.second;
    auto tcp_client = GetOrCreateTcpClient(recv_address);
    if (tcp_client == nullptr) {
      MS_LOG_WARNING << "Failed to connect to server, node id: " << recv_node << ", node tcp address: " << recv_address;
      continue;
    }
    auto request_track = AddMessageTrack(1, nullptr);
    MessageMeta message_meta;
    message_meta.set_cmd(NodeCommand::BROADCAST_MODEL_WEIGHT);
    message_meta.set_request_id(request_track->request_id());
    message_meta.set_iteration_num(iteration_num);
    message_meta.set_send_node(send_node);
    message_meta.set_recv_node(recv_node);
    message_meta.set_role(node_info_.node_role_);
    auto ret = tcp_client->SendMessage(message_meta, Protos::PROTOBUF, proto_model.data(), proto_model.size());
    if (!ret) {
      MS_LOG(WARNING) << "Get model weight from tcp server " << recv_address << " failed";
      continue;
    }
  }
  MS_LOG_INFO << "End broadcast model weight";
}

void ServerNode::HandleBroadcastModelWeight(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                            const Protos &protos, const VectorPtr &data) {
  MS_LOG_INFO << "Receive broadcast model weight message from " << meta.send_node();
  auto ret = Executor::GetInstance().OnReceiveModelWeight(data->data(), data->size());
  if (!ret) {
    MS_LOG_WARNING << "Handle broadcast model weight request failed";
  }
  conn->SimpleResponse(meta);
}

bool ServerNode::PullWeight(const uint8_t *req_data, size_t len, VectorPtr *output) {
  MS_LOG_DEBUG << "Begin pull weight from other servers";
  if (output == nullptr) {
    return false;
  }
  *output = nullptr;
  std::map<std::string, std::string> node_map;
  auto cache_ret = cache::Server::Instance().GetAllServersRealtime(&node_map);
  if (!cache_ret.IsSuccess()) {
    return false;
  }
  const auto &send_node = node_info_.node_id_;
  auto iteration_num = cache::InstanceContext::Instance().iteration_num();
  for (auto &item : node_map) {
    const auto &recv_node = item.first;
    if (send_node == recv_node) {
      continue;
    }
    const auto &recv_address = item.second;
    auto tcp_client = GetOrCreateTcpClient(recv_address);
    if (tcp_client == nullptr) {
      MS_LOG_WARNING << "Failed to connect to server, node id: " << recv_node << ", node tcp address: " << recv_address;
      continue;
    }
    auto request_track =
      AddMessageTrack(1, [output, recv_node](const MessageMeta &meta, const VectorPtr &response_data) {
        if (!meta.response_error().empty()) {
          return;
        }
        *output = response_data;
      });
    MessageMeta message_meta;
    message_meta.set_cmd(NodeCommand::SERVER_PULL_WEIGHT);
    message_meta.set_request_id(request_track->request_id());
    message_meta.set_iteration_num(iteration_num);
    message_meta.set_send_node(send_node);
    message_meta.set_recv_node(recv_node);
    message_meta.set_role(node_info_.node_role_);
    auto ret = tcp_client->SendMessage(message_meta, Protos::FLATBUFFERS, req_data, len);
    if (!ret) {
      MS_LOG(WARNING) << "Send pull weight message to tcp server " << recv_address << " failed";
      continue;
    }
    constexpr int timeout_in_seconds_wait_response = 30;
    ret = Wait(request_track, timeout_in_seconds_wait_response);
    if (!ret) {
      continue;
    }
    if (*output == nullptr) {
      continue;
    }
    MS_LOG_INFO << "Success to pull weight from other server " << recv_node << ", pull weight size is "
                << (*output)->size();
    return true;
  }
  MS_LOG_DEBUG << "End pull weight from other servers";
  return false;
}

void ServerNode::HandleServerPullWeight(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                        const Protos &protos, const VectorPtr &data) {
  MS_LOG_DEBUG << "Begin handle pull weight request from " << meta.send_node();
  FBBuilder fbb;
  auto ret = Executor::GetInstance().HandlePullWeightRequest(data->data(), data->size(), &fbb);
  if (!ret.IsSuccess()) {
    conn->ErrorResponse(meta, ret.StatusMessage());
    return;
  }
  conn->SendMessage(meta, Protos::FLATBUFFERS, fbb.GetBufferPointer(), fbb.GetSize());
  MS_LOG_DEBUG << "End handle pull weight request";
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
