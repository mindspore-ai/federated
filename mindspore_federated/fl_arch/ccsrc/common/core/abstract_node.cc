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

#include "common/core/abstract_node.h"
#include "common/communicator/tcp_communicator.h"
#include "communicator/http_communicator.h"
#include "distributed_cache/instance_context.h"
#include "common/common.h"

namespace mindspore {
namespace fl {
ResponseTrack::ResponseTrack(AbstractNode *node, uint64_t request_id, uint64_t expect_count,
                             const MessageCallback &callback)
    : node_(node), request_id_(request_id), expect_count_(expect_count), callback_(callback) {}

ResponseTrack::~ResponseTrack() {
  if (node_ != nullptr) {
    node_->ReleaseResponseTrack(request_id_);
  }
}

bool ResponseTrack::OnRecvResponseData(const MessageMeta &meta, const Protos &protos, const VectorPtr &data) {
  if (callback_ != nullptr) {
    callback_(meta, data);
  }
  curr_count_ += 1;
  return curr_count_ >= expect_count_;
}

bool ResponseTrack::OnRecvResponseData() {
  curr_count_ += 1;
  return curr_count_ >= expect_count_;
}

bool ResponseTrack::CheckMessageTrack() const { return curr_count_ >= expect_count_; }

std::shared_ptr<TcpClient> AbstractNode::GetOrCreateTcpClient(const std::string &server_address) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  for (auto it = tcp_client_map_.begin(); it != tcp_client_map_.end();) {
    auto client = it->second;
    if (!client || !client->connected()) {
      it = tcp_client_map_.erase(it);
    } else {
      ++it;
    }
  }
  auto it = tcp_client_map_.find(server_address);
  if (it != tcp_client_map_.end()) {
    auto client = it->second;
    if (client && client->connected()) {
      return it->second;
    }
  }
  std::string ip;
  uint32_t port;
  if (!CommUtil::SplitIpAddress(server_address, &ip, &port)) {
    MS_LOG(ERROR) << "Connection error: invalid server address: " << server_address;
    return nullptr;
  }
  auto client = std::make_shared<TcpClient>(ip, port, NodeRole::SERVER);
  if (client == nullptr) {
    return nullptr;
  }
  client->SetMessageCallback([&](const MessageMeta &meta, const Protos &protos, const VectorPtr &data) {
    NotifyMessageArrival(meta, protos, data);
  });
  constexpr int timeout_in_seconds_wait_connected = 3;
  auto res = client->Start(timeout_in_seconds_wait_connected);
  if (!res) {
    MS_LOG_WARNING << "Connect to tcp server " << server_address << " failed";
    return nullptr;
  }
  tcp_client_map_[server_address] = client;
  return client;
}

static std::string CollectiveMetaToString(const CollectiveMessageMeta &meta) {
  std::ostringstream os;
  os << "{iteration:" << meta.iteration() << ", data:" << meta.weight_name() << ", send rank:" << meta.send_node()
     << ", recv rank:" << meta.recv_node() << ", phase:" << meta.phase() << ", chunk index:" << meta.chunk_index()
     << ", for index:" << meta.for_index() << "}";
  return os.str();
}

std::shared_ptr<ResponseTrack> AbstractNode::CollectiveSendAsync(const std::string &recv_address,
                                                                 const CollectiveMessageMeta &collective_meta,
                                                                 const void *data, size_t size) {
  if (data == nullptr) {
    return nullptr;
  }
  const auto &recv_node = collective_meta.recv_node();
  MessageMeta message_meta;
  message_meta.set_cmd(NodeCommand::COLLECTIVE_SEND_DATA);
  message_meta.set_role(node_info_.node_role_);
  message_meta.set_recv_node(recv_node);
  message_meta.set_send_node(node_info_.node_id_);
  *(message_meta.mutable_collective_meta()) = collective_meta;
  message_meta.mutable_collective_meta()->set_enable_flag(true);
  message_meta.mutable_collective_meta()->set_send_node(node_info_.node_id_);
  auto client = GetOrCreateTcpClient(recv_address);
  if (client == nullptr) {
    return nullptr;
  }
  auto request_track = AddMessageTrack(1, nullptr);
  message_meta.set_request_id(request_track->request_id());
  auto ret = client->SendMessage(message_meta, Protos::RAW, data, size);
  if (!ret) {
    return nullptr;
  }
  return request_track;
}

bool AbstractNode::CollectiveRecvWaitInner(const CollectiveMessageMeta &expect_meta, VectorPtr *output,
                                           const uint32_t &timeout) {
  if (output == nullptr) {
    return false;
  }
  const auto &send_node = expect_meta.send_node();
  auto check_meta = [](const CollectiveMessageMeta &left, const CollectiveMessageMeta &right) {
    return left.iteration() == right.iteration() && left.weight_name() == right.weight_name() &&
           left.recv_node() == right.recv_node() && left.send_node() == right.send_node() &&
           left.phase() == right.phase() && left.chunk_index() == right.chunk_index() &&
           left.for_index() == right.for_index();
  };
  auto iteration_num = expect_meta.iteration();
  std::unique_lock<std::mutex> lock(collective_received_mutex_);
  auto &recv_data_list = collective_received_data_[send_node];
  for (uint32_t i = 0; i < timeout; i++) {
    if (recv_data_list.empty()) {
      collective_received_cond_.wait_for(lock, std::chrono::seconds(1),
                                         [&recv_data_list]() { return !recv_data_list.empty(); });
      if (recv_data_list.empty()) {  // timeout
        if (cache::InstanceContext::Instance().HasIterationFailed(iteration_num)) {
          MS_LOG(WARNING) << "Detect iteration " << iteration_num << " has failed";
          return false;
        }
        continue;
      }
    }
    while (!recv_data_list.empty()) {
      auto first = recv_data_list.begin();
      auto recv_meta = std::move(first->first);
      auto recv_data = std::move(first->second);
      recv_data_list.erase(first);
      MS_LOG(DEBUG) << "Handle receive data from node:" << send_node
                    << ", recv meta:" << CollectiveMetaToString(recv_meta);
      if (recv_meta.iteration() != expect_meta.iteration()) {
        MS_LOG(WARNING) << "Skip recv data, iteration of recv meta " << recv_meta.iteration()
                        << " != iteration of expected meta " << expect_meta.iteration();
        continue;
      }
      // error data in the same iteration
      if (!check_meta(recv_meta, expect_meta)) {
        MS_LOG(WARNING) << "Recv meta not match expected meta, recv mata: " << CollectiveMetaToString(recv_meta)
                        << ", expected meta: " << CollectiveMetaToString(expect_meta);
        return false;
      }
      *output = recv_data;
      return true;  // success to recv data
    }
  }
  return false;
}

bool AbstractNode::CollectiveRecvWait(const CollectiveMessageMeta &expect_meta, size_t expect_size, VectorPtr *output,
                                      const uint32_t &timeout) {
  if (output == nullptr) {
    MS_LOG(ERROR) << "CollectiveRecvWait failed, parameter output invalid";
    return false;
  }
  auto data_recved = CollectiveRecvWaitInner(expect_meta, output, timeout);
  if (!data_recved) {
    MS_LOG(ERROR) << "CollectiveRecvWait failed, expect meta: " << CollectiveMetaToString(expect_meta);
    return false;
  }
  if (*output == nullptr) {
    MS_LOG(ERROR) << "CollectiveRecvWait failed, recv buffer invalid";
    return false;
  }
  if (expect_size != (*output)->size()) {
    MS_LOG(ERROR) << "Expected data size " << expect_size << " != recv data size " << (*output)->size()
                  << CollectiveMetaToString(expect_meta);
    return false;
  }
  return true;
}

std::string AbstractNode::node_id() const { return node_info_.node_id_; }

std::string AbstractNode::tcp_address() const { return node_info_.ip_ + ":" + std::to_string(node_info_.port_); }

std::shared_ptr<CommunicatorBase> AbstractNode::GetOrCreateHttpComm(const std::string &ip, uint16_t port) {
  std::lock_guard<std::mutex> lock(communicator_mutex_);
  if (http_communicator_ == nullptr) {
    MS_LOG(INFO) << "Create Http communicator.";
    http_server_ = std::make_shared<HttpServer>(ip, port, kThreadNum);
    MS_EXCEPTION_IF_NULL(http_server_);
    http_communicator_ = std::make_shared<HttpCommunicator>(http_server_);
    MS_EXCEPTION_IF_NULL(http_communicator_);
  }
  return http_communicator_;
}

std::shared_ptr<CommunicatorBase> AbstractNode::GetOrCreateTcpComm() {
  std::lock_guard<std::mutex> lock(communicator_mutex_);
  if (tcp_communicator_ == nullptr) {
    MS_LOG(INFO) << "Create Tcp communicator.";
    tcp_communicator_ = std::make_shared<TcpCommunicator>();
    MS_EXCEPTION_IF_NULL(tcp_communicator_);
  }
  return tcp_communicator_;
}

void AbstractNode::StartTcpServer(const std::string &ip, uint16_t port) {
  tcp_server_ = std::make_shared<TcpServer>(ip, port);
  MS_EXCEPTION_IF_NULL(tcp_server_);
  tcp_server_->SetMessageCallback([this](const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                         const Protos &protos,
                                         const VectorPtr &data) { TcpMessageHandle(conn, meta, protos, data); });
  tcp_server_->Start();

  node_info_.ip_ = tcp_server_->BoundIp();
  node_info_.port_ = tcp_server_->BoundPort();

  // create tcp client to itself in case of event dispatch failed when there are no events pending or actvie
  tcp_client_local_ = std::make_shared<TcpClient>(node_info_.ip_, node_info_.port_, NodeRole::SERVER);
  constexpr int timeout_in_seconds_wait_connected = 3;
  tcp_client_local_->Start(timeout_in_seconds_wait_connected);
}

void AbstractNode::StartTcpCommunicator() {
  std::lock_guard<std::mutex> lock(communicator_mutex_);
  if (tcp_communicator_ == nullptr) {
    return;
  }
  tcp_communicator_->Start();
}

void AbstractNode::StopTcpServer() {
  std::lock_guard<std::mutex> lock(communicator_mutex_);
  if (tcp_server_) {
    tcp_server_->Stop();
    tcp_server_ = nullptr;
  }
  if (tcp_communicator_) {
    tcp_communicator_->Stop();
    tcp_communicator_ = nullptr;
  }
  if (tcp_client_local_) {
    tcp_client_local_->Stop();
    tcp_client_local_ = nullptr;
  }
  std::lock_guard<std::mutex> lock_client(client_mutex_);
  tcp_client_map_.clear();
}

void AbstractNode::StartHttpServer() {
  std::lock_guard<std::mutex> lock(communicator_mutex_);
  if (http_server_ == nullptr || http_communicator_ == nullptr) {
    return;
  }
  http_communicator_->Start();
  MS_LOG(INFO) << "Initialize http server IP:" << http_server_->address() << ", PORT:" << http_server_->port();
  if (!http_server_->Start()) {
    MS_LOG(EXCEPTION) << "Http server starting failed.";
  }
  MS_LOG(INFO) << "Http communicator started.";
}

void AbstractNode::StopHttpServer() {
  std::lock_guard<std::mutex> lock(communicator_mutex_);
  if (http_server_) {
    http_server_->Stop();
    http_server_ = nullptr;
  }
  if (http_communicator_) {
    http_communicator_->Stop();
    http_communicator_ = nullptr;
  }
}

void AbstractNode::ProcessRoundRequest(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                       const Protos &type, const VectorPtr &data) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(data);
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << meta.request_id()
                << " the current time is:"
                << std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now())
                     .time_since_epoch()
                     .count();
  auto tcp_communicator = tcp_communicator_;
  if (tcp_communicator == nullptr) {
    conn->ErrorResponse(meta, "Tcp communicator is not inited");
    return;
  }
  tcp_communicator->HandleRoundRequest(conn, meta, type, data);
}

void AbstractNode::HandleCollectiveData(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                        const Protos &, const VectorPtr &data) {
  MS_EXCEPTION_IF_NULL(data);
  conn->SimpleResponse(meta);
  std::unique_lock<std::mutex> lock(collective_received_mutex_);
  auto &recv_meta = meta.collective_meta();
  const auto &send_node = recv_meta.send_node();
  MS_LOG(DEBUG) << "Receive data from node:" << send_node << ", recv meta:" << CollectiveMetaToString(recv_meta);
  collective_received_data_[send_node].emplace_back(std::make_pair(recv_meta, data));
  collective_received_cond_.notify_all();
}

void AbstractNode::InitNodeInfo(const NodeRole &role) {
  constexpr int rand_range = 90000;
  constexpr int rand_min = 10000;
  auto rand_num = std::rand() % rand_range + rand_min;  // 10000~99999
  node_info_.node_id_ =
    node_info_.ip_ + ":" + std::to_string(node_info_.port_) + "::" + GetTimeString() + "::" + std::to_string(rand_num);
  node_info_.node_role_ = role;

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " is generate uuid is:" << node_info_.node_id_ << ", the ip:" << node_info_.ip_
               << ", the port:" << node_info_.port_;
}

void AbstractNode::OnIterationUpdate() {
  std::unique_lock<std::mutex> lock(collective_received_mutex_);
  collective_received_data_.clear();
}

bool AbstractNode::TcpMessageHandle(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                    const Protos &protos, const VectorPtr &data) {
  MS_EXCEPTION_IF_NULL(data);
  if (meta.recv_node() != node_id()) {
    auto error_msg = "expect recv node " + meta.recv_node() + " != actual node " + node_id();
    MS_LOG_WARNING << error_msg;
    conn->ErrorResponse(meta, error_msg);
    return false;
  }
  if (TcpMessageHandleSubclass(conn, meta, protos, data)) {
    return true;
  }
  switch (meta.cmd()) {
    case NodeCommand::COLLECTIVE_SEND_DATA:
      HandleCollectiveData(conn, meta, protos, data);
      break;
    case NodeCommand::ROUND_REQUEST:
      ProcessRoundRequest(conn, meta, protos, data);
      break;
    default:
      auto error_msg = "The cmd " + std::to_string(meta.cmd()) + " is not supported!";
      MS_LOG_WARNING << error_msg;
      conn->ErrorResponse(meta, error_msg);
      return false;
  }
  return true;
}

bool AbstractNode::Wait(const std::shared_ptr<ResponseTrack> &request_track, const uint32_t &timeout) {
  if (request_track == nullptr) {
    return false;
  }
  std::unique_lock<std::mutex> tracker_lock(message_tracker_mutex_);
  bool track_check = false;
  auto request_id = request_track->request_id();
  for (uint32_t i = 0; i < timeout; i++) {
    message_tracker_cond_.wait_for(tracker_lock, std::chrono::seconds(1), [this, request_id, &track_check] {
      auto it = message_tracker_.find(request_id);
      if (it == message_tracker_.end()) {
        return true;
      }
      auto track = it->second.lock();
      if (track == nullptr) {
        return true;
      }
      if (track->CheckMessageTrack()) {
        track_check = true;
        return true;
      }
      return false;
    });
    if (track_check) {
      break;
    }
  }
  (void)message_tracker_.erase(request_id);
  return track_check;
}

std::shared_ptr<ResponseTrack> AbstractNode::AddMessageTrack(const uint32_t &expected_response,
                                                             const ResponseTrack::MessageCallback &callback) {
  std::unique_lock<std::mutex> lock(message_tracker_mutex_);
  uint64_t request_id = ++next_request_id_;
  auto track = std::make_shared<ResponseTrack>(this, request_id, expected_response, callback);
  message_tracker_[request_id] = track;
  lock.unlock();  // release track first
  return track;
}

void AbstractNode::ReleaseResponseTrack(uint64_t request_id) {
  std::lock_guard<std::mutex> lock(message_tracker_mutex_);
  message_tracker_.erase(request_id);
}

void AbstractNode::NotifyMessageArrival(const MessageMeta &meta, const Protos &protos, const VectorPtr &data) {
  std::unique_lock<std::mutex> lock(message_tracker_mutex_);
  uint64_t request_id = meta.request_id();
  auto it = message_tracker_.find(request_id);
  if (it == message_tracker_.end()) {
    return;
  }
  auto track = it->second.lock();
  if (track == nullptr) {
    message_tracker_cond_.notify_all();
    return;
  }
  auto count_enough = track->OnRecvResponseData(meta, protos, data);
  if (count_enough) {
    message_tracker_cond_.notify_all();
  }
  lock.unlock();  // release track first
}

void AbstractNode::NotifyMessageArrival(const std::shared_ptr<ResponseTrack> &response_track) {
  std::unique_lock<std::mutex> lock(message_tracker_mutex_);
  if (message_tracker_.count(response_track->request_id()) <= 0) {
    return;
  }
  response_track->OnRecvResponseData();
  message_tracker_cond_.notify_all();
  lock.unlock();  // release track first
}
}  // namespace fl
}  // namespace mindspore
