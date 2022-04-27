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

#include "scheduler.h"

namespace mindspore {
namespace fl {
Scheduler &Scheduler::GetInstance() {
  static Scheduler instance{};
  return instance;
}

bool Scheduler::Run() {
  MS_LOG(INFO) << "Start scheduler.";
  FLContext::instance()->cluster_config().scheduler_ip = FLContext::instance()->scheduler_ip();
  FLContext::instance()->cluster_config().scheduler_port = FLContext::instance()->scheduler_port();
  FLContext::instance()->cluster_config().initial_worker_num = FLContext::instance()->initial_worker_num();
  FLContext::instance()->cluster_config().initial_server_num = FLContext::instance()->initial_server_num();
  if (!scheduler_node_->Start()) {
    MS_LOG(WARNING) << "Scheduler start failed.";
    return false;
  }

  if (!scheduler_node_->Finish()) {
    MS_LOG(WARNING) << "Scheduler finish failed.";
    return false;
  }

  if (!scheduler_node_->Stop()) {
    MS_LOG(WARNING) << "Scheduler stop failed.";
    return false;
  }
  return true;
}
}  // namespace fl
}  // namespace mindspore
