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

#ifndef MINDSPORE_CCSRC_FL_SERVER_SERVER_H_
#define MINDSPORE_CCSRC_FL_SERVER_SERVER_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include "communicator/communicator_base.h"
#include "communicator/tcp_communicator.h"
#include "armour/cipher/cipher_init.h"
#include "common/common.h"
#include "common/exit_handler.h"
#include "server/executor.h"
#include "server/iteration.h"
#include "server/server_node.h"
#include "common/utils/python_adapter.h"

namespace mindspore {
namespace fl {
namespace server {
struct FlCallback {
  std::function<void()> after_started = nullptr;
  std::function<void()> before_stopped = nullptr;
  std::function<void()> on_iteration_end = nullptr;
};

// The sleeping time of the server thread before the networking is completed.
constexpr uint32_t kServerSleepTimeForNetworking = 1000;
constexpr uint64_t kDefaultReplayAttackTimeDiff = 600000;
// Class Server is the entrance of MindSpore's parameter server training mode and federated learning.
class MS_EXPORT Server {
 public:
  static Server &GetInstance();
  // According to the current MindSpore framework, method Run is a step of the server pipeline. This method will be
  // blocked until the server is finalized.
  // func_graph is the frontend graph which will be parse in server's executor and aggregator.

  // Each step of the server pipeline may have dependency on other steps, which includes:
  // InitServerContext must be the first step to set contexts for later steps.

  // Server Running relies on URL or Message Type Register:
  // StartCommunicator---->InitIteration

  // Metadata Register relies on Hash Ring of Servers which relies on Network Building Completion:
  // RegisterRoundKernel---->StartCommunicator

  // Kernel Initialization relies on Executor Initialization:
  // RegisterRoundKernel---->InitExecutor

  // Getting Model Size relies on ModelStorage Initialization which relies on Executor Initialization:
  // InitCipher---->InitExecutor
  void Run(const std::vector<InputWeight> &feature_map, const uint64_t &recovery_iteration,
           const FlCallback &fl_callback);

  void BroadcastModelWeight(const std::string &proto_model,
                            const std::map<std::string, std::string> &broadcast_server_map = {});
  bool PullWeight(const uint8_t *req_data, size_t len, VectorPtr *output);

 private:
  Server() = default;
  ~Server();
  Server(const Server &) = delete;
  Server &operator=(const Server &) = delete;

  void InitRoundConfigs();
  // Load variables which is set by ps_context.
  void InitServerContext();

  void InitServer();
  // Initialize the server cluster, server node and communicators.
  void InitCluster();
  bool InitCommunicatorWithServer();
  bool InitCommunicatorWithWorker();

  // Initialize iteration with rounds. Which rounds to use could be set by ps_context as well.
  void InitIteration();

  // Initialize executor according to the server mode.
  void InitExecutor(const std::vector<InputWeight> &init_feature_map);

  void PingOtherServers();
  void RegisterServer();
  void LockCache();
  void UnlockCache();

  void SyncAndCheckModelInfo(const std::vector<InputWeight> &init_feature_map);
  FlStatus SyncAndInitModel(const std::vector<InputWeight> &init_feature_map);

  // Initialize cipher according to the public param.
  void InitCipher();

  void InitAndLoadDistributedCache(uint64_t recovery_iteration);

  // The communicators should be started after all initializations are completed.
  void StartCommunicator();

  void RunMainProcess();

  // load pki huks cbg root certificate and crl
  void InitPkiCertificate();
  void RunMainProcessInner();
  void Stop();
  void CallIterationEndCallback();
  void CallServerStartedCallback();
  void CallServerStoppedCallback();

  // The server node is initialized in Server.
  std::shared_ptr<ServerNode> server_node_ = nullptr;

  // The configuration of all rounds.
  std::vector<RoundConfig> rounds_config_;
  armour::CipherConfig cipher_config_;
  armour::CipherInit *cipher_init_ = nullptr;

  // Server need a tcp communicator to communicate with other servers for counting, metadata storing, collective
  // operations, etc.
  std::shared_ptr<CommunicatorBase> communicator_with_server_ = nullptr;

  // The communication with workers(including mobile devices), has multiple protocol types: HTTP and TCP.
  // In some cases, both types should be supported in one distributed training job. So here we may have multiple
  // communicators.
  std::vector<std::shared_ptr<CommunicatorBase>> communicators_with_worker_;
  bool has_stopped_ = false;

  FlCallback fl_callback_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_SERVER_H_
