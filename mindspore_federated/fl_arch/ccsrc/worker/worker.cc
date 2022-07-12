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

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "common/exit_handler.h"
#include "worker/worker.h"
#include "armour/secure_protocol/key_agreement.h"
#include "distributed_cache/distributed_cache.h"
#include "distributed_cache/worker.h"

namespace mindspore {
namespace fl {
namespace worker {
Worker &Worker::GetInstance() {
  static Worker instance;
  return instance;
}

void Worker::Init() {
  if (running_.load()) {
    MS_LOG_EXCEPTION << "Worker has been inited";
  }
  running_ = true;
  ExitHandler::Instance().InitSignalHandle();

  InitAndLoadDistributedCache();
  worker_node_ = std::make_shared<fl::WorkerNode>();
  MS_EXCEPTION_IF_NULL(worker_node_);

  if (!worker_node_->Start()) {
    MS_LOG(EXCEPTION) << "Starting worker node failed.";
  }
  StartPeriodJob();
}

void Worker::StartPeriodJob() {
  cache::Worker::Instance().Init(fl_id(), FLContext::instance()->fl_name());
  auto status = cache::Worker::Instance().Register();
  if (!status.IsSuccess()) {
    MS_LOG_EXCEPTION << "Failed to register worker to distributed cache.";
  }
  MS_LOG_INFO << "Register worker to distributed cache successfully";
  auto period_fun = []() {
    auto cache_link_valid = true;
    while (!ExitHandler::Instance().HasStopped()) {
      constexpr int default_sync_duration_ms = 1000;  // 1000ms
      std::this_thread::sleep_for(std::chrono::milliseconds(default_sync_duration_ms));

      auto cache_ret = cache::Worker::Instance().Sync();
      if (!cache_ret.IsSuccess()) {
        constexpr int64_t log_interval = 15000;  // 15s
        static int64_t last_retry_timestamp_ms = 0;
        if (cache_link_valid) {
          cache_link_valid = false;
          last_retry_timestamp_ms = 0;
        }
        int64_t current_timestamp_ms = CURRENT_TIME_MILLI.count();
        if (current_timestamp_ms - last_retry_timestamp_ms >= log_interval) {
          last_retry_timestamp_ms = current_timestamp_ms;
          MS_LOG_ERROR << "Failed to reconnect to distributed cache: " << cache_ret.GetDetail();
        }
      } else {
        if (!cache_link_valid) {
          cache_link_valid = true;
          MS_LOG_INFO << "Success to reconnect to distributed cache";
        }
      }
    }
  };
  period_thread_ = std::thread(period_fun);
}

void Worker::Stop() {
  MS_LOG_INFO << "Start to stop worker";
  ExitHandler::Instance().SetStopFlag();
  if (period_thread_.joinable()) {
    period_thread_.join();
  }
  cache::Worker::Instance().Stop();
  MS_LOG_INFO << "Stop worker successfully";
}

void Worker::InitAndLoadDistributedCache() {
  auto config = FLContext::instance()->distributed_cache_config();
  if (config.address.empty()) {
    MS_LOG(EXCEPTION) << "Distributed cache address cannot be empty.";
  }
  if (!cache::DistributedCacheLoader::Instance().InitCacheImpl(config)) {
    MS_LOG(EXCEPTION) << "Link to distributed cache failed, distributed cache address: " << config.address
                      << ", enable ssl: " << config.enable_ssl;
  }
}

bool Worker::SendToServer(const void *data, size_t size, TcpUserCommand command, VectorPtr *output) {
  MS_EXCEPTION_IF_NULL(worker_node_);
  MS_EXCEPTION_IF_NULL(data);
  if (output != nullptr) {
    while (!ExitHandler::Instance().HasStopped()) {
      if (!worker_node_->Send(NodeRole::SERVER, data, size, static_cast<int>(command), output, kWorkerTimeout)) {
        MS_LOG(ERROR) << "Sending message to server failed.";
        return false;
      }
      if (*output == nullptr) {
        MS_LOG(WARNING) << "Response from server is empty.";
        return false;
      }

      std::string response_str = std::string(reinterpret_cast<char *>((*output)->data()), (*output)->size());
      if (response_str == kClusterSafeMode || response_str == kJobNotAvailable) {
        MS_LOG(INFO) << "The server is in safe mode, or is disabled or finished.";
        std::this_thread::sleep_for(std::chrono::milliseconds(kWorkerRetryDurationForSafeMode));
      } else {
        break;
      }
    }
  } else {
    if (!worker_node_->Send(NodeRole::SERVER, data, size, static_cast<int>(command), nullptr, kWorkerTimeout)) {
      MS_LOG(ERROR) << "Sending message to server failed.";
      return false;
    }
  }
  return true;
}

void Worker::SetIterationRunning() {
  MS_LOG(INFO) << "Worker iteration starts.";
  worker_iteration_state_ = IterationState::kRunning;
}

void Worker::SetIterationCompleted() {
  MS_LOG(INFO) << "Worker iteration completes.";
  worker_iteration_state_ = IterationState::kCompleted;
}

void Worker::set_fl_iteration_num(uint64_t iteration_num) { iteration_num_ = iteration_num; }

uint64_t Worker::fl_iteration_num() const { return iteration_num_.load(); }

void Worker::set_data_size(int data_size) { data_size_ = data_size; }

void Worker::set_secret_pk(armour::PrivateKey *secret_pk) { secret_pk_ = secret_pk; }

void Worker::set_pw_salt(const std::vector<uint8_t> &pw_salt) { pw_salt_ = pw_salt; }

void Worker::set_pw_iv(const std::vector<uint8_t> &pw_iv) { pw_iv_ = pw_iv; }

void Worker::set_public_keys_list(const std::vector<EncryptPublicKeys> &public_keys_list) {
  public_keys_list_ = public_keys_list;
}

int Worker::data_size() const { return data_size_; }

armour::PrivateKey *Worker::secret_pk() const { return secret_pk_; }

std::vector<uint8_t> Worker::pw_salt() const { return pw_salt_; }

std::vector<uint8_t> Worker::pw_iv() const { return pw_iv_; }

std::vector<EncryptPublicKeys> Worker::public_keys_list() const { return public_keys_list_; }

std::string Worker::fl_name() const { return kServerModeFL; }

std::string Worker::fl_id() const {
  MS_EXCEPTION_IF_NULL(worker_node_);
  return worker_node_->fl_id();
}

std::string Worker::encrypt_type() const { return encrypt_type_; }
}  // namespace worker
}  // namespace fl
}  // namespace mindspore
