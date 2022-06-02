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
#include "worker/fl_worker.h"
#include "armour/secure_protocol/key_agreement.h"

namespace mindspore {
namespace fl {
namespace worker {
FLWorker &FLWorker::GetInstance() {
  static FLWorker instance;
  return instance;
}

bool FLWorker::Run() {
  if (running_.load()) {
    return false;
  }
  running_ = true;
  worker_num_ = FLContext::instance()->initial_worker_num();
  server_num_ = FLContext::instance()->initial_server_num();
  scheduler_ip_ = FLContext::instance()->scheduler_ip();
  scheduler_port_ = FLContext::instance()->scheduler_port();
  worker_step_num_per_iteration_ = FLContext::instance()->worker_step_num_per_iteration();
  encrypt_type_ = FLContext::instance()->encrypt_type();
  FLContext::instance()->cluster_config().scheduler_ip = scheduler_ip_;
  FLContext::instance()->cluster_config().scheduler_port = scheduler_port_;
  FLContext::instance()->cluster_config().initial_worker_num = worker_num_;
  FLContext::instance()->cluster_config().initial_server_num = server_num_;
  MS_LOG(INFO) << "Initialize cluster config for worker. Worker number:" << worker_num_
               << ", Server number:" << server_num_ << ", Scheduler ip:" << scheduler_ip_
               << ", Scheduler port:" << scheduler_port_
               << ", Worker training step per iteration:" << worker_step_num_per_iteration_ << ", Encrypt type: " << encrypt_type_;

  worker_node_ = std::make_shared<fl::core::WorkerNode>();
  MS_EXCEPTION_IF_NULL(worker_node_);

  worker_node_->SetCancelSafeModeCallBack([this]() -> void { safemode_ = false; });
  worker_node_->RegisterEventCallback(fl::core::ClusterEvent::SCHEDULER_TIMEOUT, [this]() {
    Finalize();
    running_ = false;
    try {
      MS_LOG(EXCEPTION)
        << "Event SCHEDULER_TIMEOUT is captured. This is because scheduler node is finalized or crashed.";
    } catch (std::exception &e) {
      MS_LOG(ERROR) << "Catch exception: " << e.what();
    }
  });
  worker_node_->RegisterEventCallback(fl::core::ClusterEvent::NODE_TIMEOUT, [this]() {
    Finalize();
    running_ = false;
    try {
      MS_LOG(EXCEPTION)
        << "Event NODE_TIMEOUT is captured. This is because some server nodes are finalized or crashed after the "
           "network building phase.";
    } catch (std::exception &e) {
      MS_LOG(ERROR) << "Catch exception: " << e.what();
    }
  });

  InitializeFollowerScaler();
  if (!worker_node_->Start()) {
    MS_LOG(EXCEPTION) << "Starting worker node failed.";
    return false;
  }
  rank_id_ = worker_node_->rank_id();

  std::this_thread::sleep_for(std::chrono::milliseconds(kWorkerSleepTimeForNetworking));
  return true;
}

void FLWorker::Finalize() {
  if (worker_node_ == nullptr) {
    MS_LOG(INFO) << "The worker is not initialized yet.";
    return;
  }

  // In some cases, worker calls the Finish function while other nodes don't. So timeout is acceptable.
  if (!worker_node_->Finish()) {
    MS_LOG(WARNING) << "Finishing worker node timeout.";
  }
  if (!worker_node_->Stop()) {
    MS_LOG(ERROR) << "Stopping worker node failed.";
    return;
  }
}

bool FLWorker::SendToServer(uint32_t server_rank, const void *data, size_t size, fl::core::TcpUserCommand command,
                            std::shared_ptr<std::vector<unsigned char>> *output) {
  MS_EXCEPTION_IF_NULL(data);
  // If the worker is in safemode, do not communicate with server.
  while (safemode_.load()) {
    std::this_thread::yield();
  }
  if (output != nullptr) {
    while (true) {
      if (!worker_node_->Send(fl::core::NodeRole::SERVER, server_rank, data, size, static_cast<int>(command), output,
                              kWorkerTimeout)) {
        MS_LOG(ERROR) << "Sending message to server " << server_rank << " failed.";
        return false;
      }
      if (*output == nullptr) {
        MS_LOG(WARNING) << "Response from server " << server_rank << " is empty.";
        return false;
      }

      std::string response_str = std::string(reinterpret_cast<char *>((*output)->data()), (*output)->size());
      if (response_str == kClusterSafeMode || response_str == kJobNotAvailable) {
        MS_LOG(INFO) << "The server " << server_rank << " is in safemode or finished.";
        std::this_thread::sleep_for(std::chrono::milliseconds(kWorkerRetryDurationForSafeMode));
      } else {
        break;
      }
    }
  } else {
    if (!worker_node_->Send(fl::core::NodeRole::SERVER, server_rank, data, size, static_cast<int>(command), nullptr,
                            kWorkerTimeout)) {
      MS_LOG(ERROR) << "Sending message to server " << server_rank << " failed.";
      return false;
    }
  }
  return true;
}

uint32_t FLWorker::server_num() const { return server_num_; }

uint32_t FLWorker::worker_num() const { return worker_num_; }

uint32_t FLWorker::rank_id() const { return rank_id_; }

uint64_t FLWorker::worker_step_num_per_iteration() const { return worker_step_num_per_iteration_; }

bool FLWorker::running() const { return running_.load(); }

void FLWorker::SetIterationRunning() {
  MS_LOG(INFO) << "Worker iteration starts.";
  worker_iteration_state_ = IterationState::kRunning;
}

void FLWorker::SetIterationCompleted() {
  MS_LOG(INFO) << "Worker iteration completes.";
  worker_iteration_state_ = IterationState::kCompleted;
}

void FLWorker::set_fl_iteration_num(uint64_t iteration_num) { iteration_num_ = iteration_num; }

uint64_t FLWorker::fl_iteration_num() const { return iteration_num_.load(); }

void FLWorker::set_data_size(int data_size) { data_size_ = data_size; }

void FLWorker::set_secret_pk(armour::PrivateKey *secret_pk) { secret_pk_ = secret_pk; }

void FLWorker::set_pw_salt(const std::vector<uint8_t> pw_salt) { pw_salt_ = pw_salt; }

void FLWorker::set_pw_iv(const std::vector<uint8_t> pw_iv) { pw_iv_ = pw_iv; }

void FLWorker::set_public_keys_list(const std::vector<EncryptPublicKeys> public_keys_list) {
  public_keys_list_ = public_keys_list;
}

int FLWorker::data_size() const { return data_size_; }

armour::PrivateKey *FLWorker::secret_pk() const { return secret_pk_; }

std::vector<uint8_t> FLWorker::pw_salt() const { return pw_salt_; }

std::vector<uint8_t> FLWorker::pw_iv() const { return pw_iv_; }

std::vector<EncryptPublicKeys> FLWorker::public_keys_list() const { return public_keys_list_; }

std::string FLWorker::fl_name() const { return kServerModeFL; }

std::string FLWorker::fl_id() const { return std::to_string(rank_id_); }

std::string FLWorker::encrypt_type() const { return encrypt_type_; }

void FLWorker::InitializeFollowerScaler() {
  MS_EXCEPTION_IF_NULL(worker_node_);
  if (!worker_node_->InitFollowerScaler()) {
    MS_LOG(EXCEPTION) << "Initializing follower elastic scaler failed.";
    return;
  }

  // Set scaling barriers before scaling.
  worker_node_->RegisterFollowerScalerBarrierBeforeScaleOut("WorkerPipeline",
                                                            std::bind(&FLWorker::ProcessBeforeScalingOut, this));
  worker_node_->RegisterFollowerScalerBarrierBeforeScaleIn("WorkerPipeline",
                                                           std::bind(&FLWorker::ProcessBeforeScalingIn, this));

  // Set handlers after scheduler scaling operations are done.
  worker_node_->RegisterFollowerScalerHandlerAfterScaleOut("WorkerPipeline",
                                                           std::bind(&FLWorker::ProcessAfterScalingOut, this));
  worker_node_->RegisterFollowerScalerHandlerAfterScaleIn("WorkerPipeline",
                                                          std::bind(&FLWorker::ProcessAfterScalingIn, this));
  worker_node_->RegisterCustomEventCallback(static_cast<uint32_t>(UserDefineEvent::kIterationRunning),
                                            std::bind(&FLWorker::HandleIterationRunningEvent, this));
  worker_node_->RegisterCustomEventCallback(static_cast<uint32_t>(UserDefineEvent::kIterationCompleted),
                                            std::bind(&FLWorker::HandleIterationCompletedEvent, this));
}

void FLWorker::HandleIterationRunningEvent() {
  MS_LOG(INFO) << "Server iteration starts, safemode is " << safemode_.load();
  server_iteration_state_ = IterationState::kRunning;
  if (safemode_.load() == true) {
    safemode_ = false;
  }
}

void FLWorker::HandleIterationCompletedEvent() {
  MS_LOG(INFO) << "Server iteration completes";
  server_iteration_state_ = IterationState::kCompleted;
}

void FLWorker::ProcessBeforeScalingOut() {
  MS_LOG(INFO) << "Starting Worker scaling out barrier.";
  while (server_iteration_state_.load() != IterationState::kCompleted ||
         worker_iteration_state_.load() != IterationState::kCompleted) {
    std::this_thread::yield();
  }
  MS_LOG(INFO) << "Ending Worker scaling out barrier. Switch to safemode.";
  safemode_ = true;
}

void FLWorker::ProcessBeforeScalingIn() {
  MS_LOG(INFO) << "Starting Worker scaling in barrier.";
  while (server_iteration_state_.load() != IterationState::kCompleted ||
         worker_iteration_state_.load() != IterationState::kCompleted) {
    std::this_thread::yield();
  }
  MS_LOG(INFO) << "Ending Worker scaling in barrier. Switch to safemode.";
  safemode_ = true;
}

void FLWorker::ProcessAfterScalingOut() {
  MS_ERROR_IF_NULL_WO_RET_VAL(worker_node_);
  MS_LOG(INFO) << "Cluster scaling out completed. Reinitialize for worker.";
  server_num_ = worker_node_->server_num();
  worker_num_ = worker_node_->worker_num();
  MS_LOG(INFO) << "After scheduler scaling out, worker number is " << worker_num_ << ", server number is "
               << server_num_ << ". Exit safemode.";
  std::this_thread::sleep_for(std::chrono::milliseconds(kWorkerSleepTimeForNetworking));
  safemode_ = false;
}

void FLWorker::ProcessAfterScalingIn() {
  MS_ERROR_IF_NULL_WO_RET_VAL(worker_node_);
  MS_LOG(INFO) << "Cluster scaling in completed. Reinitialize for worker.";
  server_num_ = worker_node_->server_num();
  worker_num_ = worker_node_->worker_num();
  MS_LOG(INFO) << "After scheduler scaling in, worker number is " << worker_num_ << ", server number is " << server_num_
               << ". Exit safemode.";
  std::this_thread::sleep_for(std::chrono::milliseconds(kWorkerSleepTimeForNetworking));
  safemode_ = false;
}
}  // namespace worker
}  // namespace fl
}  // namespace mindspore