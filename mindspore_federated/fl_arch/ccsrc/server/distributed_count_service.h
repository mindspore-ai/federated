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

#ifndef MINDSPORE_CCSRC_FL_SERVER_DISTRIBUTED_COUNT_SERVICE_H_
#define MINDSPORE_CCSRC_FL_SERVER_DISTRIBUTED_COUNT_SERVICE_H_

#include <set>
#include <string>
#include <memory>
#include <unordered_map>
#include "common/common.h"
#include "communicator/tcp_communicator.h"
#include "server/server_node.h"

namespace mindspore {
namespace fl {
namespace server {
// The callbacks for the first count and last count event.
using CounterCallback = std::function<void()>;

// DistributedCountService is used for counting in the server cluster dimension. It's used for counting of rounds,
// aggregation counting, etc.

// The counting could be called by any server, but only one server has the information
// of the cluster count and we mark this server as the counting server. Other servers must communicate with this
// counting server to increase/query count number.

// On the first count or last count event, DistributedCountService on the counting server triggers the event on other
// servers by sending counter event commands. This is for the purpose of keeping server cluster's consistency.
class DistributedCountService {
 public:
  static DistributedCountService &GetInstance() {
    static DistributedCountService instance;
    return instance;
  }

  // Initialize counter service with the server node because communication is needed.
  void Initialize(const std::shared_ptr<ServerNode> &server_node);

  // Register counter to the counting server for the name with its threshold count in server cluster dimension and
  // first/last count event callbacks.
  void RegisterCounter(const std::string &name, int64_t global_threshold_count, const CounterCallback &first_callback,
                       const CounterCallback &last_callback);

  // Report a count to the counting server. Parameter 'id' is in case of repeated counting. Parameter 'reason' is the
  // reason why counting failed.
  bool Count(const std::string &name);

  // Query whether the count reaches the threshold count for the name. If the count is the same as the threshold count,
  // this method returns true.
  bool CountReachThreshold(const std::string &name);

 private:
  std::shared_ptr<ServerNode> server_node_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_DISTRIBUTED_COUNT_SERVICE_H_
