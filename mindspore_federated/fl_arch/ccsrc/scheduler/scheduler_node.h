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

#ifndef MINDSPORE_CCSRC_FL_SCHEDULER_NODE_H_
#define MINDSPORE_CCSRC_FL_SCHEDULER_NODE_H_

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <condition_variable>

#include "common/fl_context.h"
#include "common/constants.h"
#include "communicator/http_message_handler.h"
#include "communicator/http_server.h"
#include "distributed_cache/instance_context.h"

namespace mindspore {
namespace fl {
class SchedulerNode {
 public:
  SchedulerNode() {}
  ~SchedulerNode();

  bool Start(const uint32_t &timeout = FLContext::instance()->cluster_config().cluster_available_timeout);
  bool Stop();

 protected:
  void Initialize();

  // Handle the get cluster state http request Synchronously.
  void ProcessGetClusterState(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the new instance http request Synchronously.
  void ProcessNewInstance(const std::shared_ptr<HttpMessageHandler> &resp);
  void ProcessQueryInstance(const std::shared_ptr<HttpMessageHandler> &resp);
  // Handle the enable FLS http request Synchronously.
  void ProcessEnableFLS(const std::shared_ptr<HttpMessageHandler> &resp);

  // Handle the disable FLS http request Synchronously.
  void ProcessDisableFLS(const std::shared_ptr<HttpMessageHandler> &resp);

  void ProcessStopFLS(const std::shared_ptr<HttpMessageHandler> &resp);

  void StartRestfulServer(const std::string &address, std::uint16_t port, size_t thread_num = 10);

  void StopRestfulServer();

  FlStatus GetClusterState(const std::string &fl_name, cache::InstanceState *state);
  FlStatus GetNodesInfoCommon(const std::string &fl_name, nlohmann::json *js);

  static void StopThreadFunc();

  std::shared_ptr<HttpServer> http_server_;
  std::unordered_map<std::string, OnRequestReceive> callbacks_;
  std::thread stop_thread_;
  bool stopping_ = false;
  std::mutex lock_;
};
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FL_SCHEDULER_NODE_H_
