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

#ifndef MINDSPORE_CCSRC_FL_WORKER_HYBRID_WORKER_H_
#define MINDSPORE_CCSRC_FL_WORKER_HYBRID_WORKER_H_

#include <memory>
#include <string>
#include <vector>
#include "common/protos/comm.pb.h"
#include "schema/fl_job_generated.h"
#include "schema/cipher_generated.h"
#include "armour/secure_protocol/key_agreement.h"
#include "common/fl_context.h"
#include "worker/worker_node.h"
#include "common/communicator/tcp_communicator.h"
#include "common/common.h"

namespace mindspore {
namespace fl {
using FBBuilder = flatbuffers::FlatBufferBuilder;

// The sleeping time of the worker thread before the networking is completed.
constexpr uint32_t kWorkerSleepTimeForNetworking = 1000;

// The time duration between retrying when server is in safemode.
constexpr uint32_t kWorkerRetryDurationForSafeMode = 500;

// The timeout for worker sending message to server in case of network jitter.
constexpr uint32_t kWorkerTimeout = 30;

enum class IterationState {
  // This iteration is still in process.
  kRunning,
  // This iteration is completed and the next iteration is not started yet.
  kCompleted
};

namespace worker {
// This class is used for hybrid training mode for now. In later version, parameter server mode will also use this class
// as worker.
class MS_EXPORT HybridWorker {
 public:
  static HybridWorker &GetInstance();
  void Init();
  void Stop();
  bool SendToServer(const void *data, size_t size, fl::TcpUserCommand command, VectorPtr *output = nullptr);

  // These methods set the worker's iteration state.
  void SetIterationRunning();
  void SetIterationCompleted();

  void set_fl_iteration_num(uint64_t iteration_num);
  uint64_t fl_iteration_num() const;

  void set_data_size(int data_size);
  int data_size() const;

  std::string fl_name() const;
  std::string fl_id() const;
  std::string instance_name() const;

 private:
  HybridWorker() = default;
  ~HybridWorker() = default;
  HybridWorker(const HybridWorker &) = delete;
  HybridWorker &operator=(const HybridWorker &) = delete;
  void InitAndLoadDistributedCache();
  void StartPeriodJob();

  std::thread period_thread_;
  std::atomic_bool running_ = false;
  std::shared_ptr<fl::WorkerNode> worker_node_ = nullptr;

  // The federated learning iteration number.
  std::atomic<uint64_t> iteration_num_ = 0;

  // Data size for this federated learning job.
  int data_size_ = 0;

  // This variable represents the worker iteration state and should be changed by worker training process.
  std::atomic<IterationState> worker_iteration_state_;
  std::string fl_name_;
  std::string instance_name_;
};
}  // namespace worker
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_WORKER_HYBRID_WORKER_H_
