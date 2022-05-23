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

#include "server/distributed_count_service.h"
#include <string>
#include <memory>
#include "distributed_cache/counter.h"

namespace mindspore {
namespace fl {
namespace server {
void DistributedCountService::Initialize(const std::shared_ptr<ServerNode> &server_node) {
  MS_EXCEPTION_IF_NULL(server_node);
  server_node_ = server_node;
}

void DistributedCountService::RegisterCounter(const std::string &name, int64_t global_threshold_count,
                                              const CounterCallback &first_callback,
                                              const CounterCallback &last_callback) {
  cache::Counter::Instance().RegisterCounter(name, global_threshold_count, first_callback, last_callback);
}

bool DistributedCountService::Count(const std::string &name) {
  bool trigger_first = false;
  bool trigger_last = false;
  auto ret = cache::Counter::Instance().Count(name, &trigger_first, &trigger_last);
  if (!ret) {
    return false;
  }
  if (trigger_first || trigger_last) {
    if (server_node_ == nullptr) {
      return true;
    }
    ServerBroadcastMessage msg;
    msg.set_type(ServerBroadcastMessage_BroadcastEventType_COUNT_EVENT);
    msg.set_count_name(name);
    msg.set_trigger_first(trigger_first);
    msg.set_trigger_last(trigger_last);
    server_node_->BroadcastEvent(msg);
  }
  return true;
}

bool DistributedCountService::CountReachThreshold(const std::string &name) {
  return cache::Counter::Instance().ReachThreshold(name);
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
