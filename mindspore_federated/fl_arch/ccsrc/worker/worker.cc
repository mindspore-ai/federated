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
#include "worker/worker.h"
#include "armour/secure_protocol/key_agreement.h"
#include "distributed_cache/distributed_cache.h"

namespace mindspore {
namespace fl {
namespace worker {
// The handler to capture the signal of SIGTERM. Normally this signal is triggered by cloud cluster manager like K8S.
namespace {
int g_signal = 0;
}
void SignalHandler(int signal) {
  if (g_signal == 0) {
    g_signal = signal;
    Worker::GetInstance().SetStopFlag();
  }
}

void Worker::SetStopFlag() { stop_flag_ = true; }
bool Worker::HasStopped() const { return stop_flag_; }

Worker &Worker::GetInstance() {
  static Worker instance;
  return instance;
}

void Worker::Init() {
  if (running_.load()) {
    MS_LOG_EXCEPTION << "Worker has been inited";
  }
  running_ = true;
  (void)signal(SIGTERM, SignalHandler);
  (void)signal(SIGINT, SignalHandler);

  InitAndLoadDistributedCache();
  worker_node_ = std::make_shared<fl::WorkerNode>();
  MS_EXCEPTION_IF_NULL(worker_node_);

  if (!worker_node_->Start()) {
    MS_LOG(EXCEPTION) << "Starting worker node failed.";
  }
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

bool Worker::SendToServer(const void *data, size_t size, TcpUserCommand command,
                          std::shared_ptr<std::vector<unsigned char>> *output) {
  MS_EXCEPTION_IF_NULL(worker_node_);
  MS_EXCEPTION_IF_NULL(data);
  if (output != nullptr) {
    while (!Worker::GetInstance().HasStopped()) {
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
