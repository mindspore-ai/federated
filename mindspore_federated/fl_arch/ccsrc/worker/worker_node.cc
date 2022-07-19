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

#include "worker/worker_node.h"
#include <map>
#include "distributed_cache/distributed_cache.h"
#include "distributed_cache/scheduler.h"

namespace mindspore {
namespace fl {
bool WorkerNode::Start(const uint32_t &timeout) {
  MS_LOG(INFO) << "[Worker start]: 1. Begin to start worker node!";
  Initialize();
  MS_LOG(INFO) << "[Worker start]: 2. Successfully start worker node!";
  return true;
}

void WorkerNode::Initialize() {
  InitNodeInfo(NodeRole::WORKER);
  constexpr int rand_range = 90000;
  constexpr int rand_min = 10000;
  auto rand_num = std::rand() % rand_range + rand_min;  // 10000~99999
  fl_id_ = "worker_fl_" + GetTimeString() + "::" + std::to_string(rand_num);
}

bool WorkerNode::Stop() { return true; }

bool WorkerNode::Send(const NodeRole &node_role, const void *message, size_t len, int command, VectorPtr *output,
                      const uint32_t &timeout) {
  std::map<std::string, std::string> server_map;
  auto fl_name = FLContext::instance()->fl_name();
  auto cache_ret = cache::Scheduler::Instance().GetAllServersRealtime(fl_name, &server_map);
  if (cache_ret.IsNil()) {
    MS_LOG_WARNING << "Cannot find fl " << fl_name << " instance info from distributed cache";
    return false;
  }
  if (!cache_ret.IsSuccess()) {
    MS_LOG_WARNING << "Failed to access distributed cache";
    return false;
  }
  if (server_map.empty()) {
    MS_LOG_WARNING << "No available server is found in distributed cache";
    return false;
  }
  MessageMeta meta;
  meta.set_cmd(NodeCommand::ROUND_REQUEST);
  meta.set_user_cmd(command);
  meta.set_role(NodeRole::WORKER);
  for (auto &server : server_map) {
    auto recv_node = server.first;
    auto recv_address = server.second;
    auto tcp_client = GetOrCreateTcpClient(recv_address);
    if (tcp_client == nullptr) {
      continue;
    }
    meta.set_recv_node(recv_node);
    ResponseTrack::MessageCallback callback = nullptr;
    if (output != nullptr) {
      callback = [output](const MessageMeta &meta, const VectorPtr &response_data) { *output = response_data; };
    }
    auto request_track = AddMessageTrack(1, callback);
    meta.set_request_id(request_track->request_id());
    if (!tcp_client->SendMessage(meta, Protos::RAW, message, len)) {
      MS_LOG_INFO << "Failed to send message to server " << recv_node;
      continue;
    }
    return Wait(request_track, timeout);
  }
  MS_LOG_WARNING << "Cannot find one server to send message";
  return false;
}
}  // namespace fl
}  // namespace mindspore
