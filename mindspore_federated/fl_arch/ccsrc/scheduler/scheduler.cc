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

#include "scheduler/scheduler.h"

namespace mindspore {
namespace fl {
namespace {
int g_signal = 0;
}
void SignalHandler(int signal) {
  if (g_signal == 0) {
    g_signal = signal;
    Scheduler::GetInstance().SetStopFlag();
  }
}

void Scheduler::SetStopFlag() { stop_flag_ = true; }

Scheduler &Scheduler::GetInstance() {
  static Scheduler instance{};
  return instance;
}

void Scheduler::Run() {
  MS_LOG(INFO) << "Start scheduler.";
  (void)signal(SIGTERM, SignalHandler);
  (void)signal(SIGINT, SignalHandler);
  InitAndLoadDistributedCache();
  scheduler_node_ = std::make_unique<SchedulerNode>();
  if (!scheduler_node_->Start()) {
    MS_LOG(EXCEPTION) << "Scheduler start failed.";
  }
  MS_LOG(INFO) << "Scheduler started successfully.";
  constexpr auto time_sleep = std::chrono::seconds(1);
  while (!stop_flag_) {
    std::this_thread::sleep_for(time_sleep);
  }
  if (!scheduler_node_->Stop()) {
    MS_LOG(WARNING) << "Scheduler stop failed.";
  }
}

void Scheduler::InitAndLoadDistributedCache() {
  auto config = FLContext::instance()->distributed_cache_config();
  if (config.address.empty()) {
    MS_LOG(EXCEPTION) << "Distributed cache address cannot be empty.";
  }
  if (!cache::DistributedCacheLoader::Instance().InitCacheImpl(config)) {
    MS_LOG(EXCEPTION) << "Link to distributed cache failed, distributed cache address: " << config.address
                      << ", enable ssl: " << config.enable_ssl;
  }
}
}  // namespace fl
}  // namespace mindspore
