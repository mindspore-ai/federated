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

#ifndef MINDSPORE_CCSRC_FL_SERVER_ITERATION_H_
#define MINDSPORE_CCSRC_FL_SERVER_ITERATION_H_

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <future>
#include "communicator/communicator_base.h"
#include "common/common.h"
#include "server/round.h"
#include "server/local_meta_store.h"
#include "server/iteration_metrics.h"
#include "server/server_node.h"

namespace mindspore {
namespace fl {
namespace server {
class IterationMetrics;
// In server's logic, Iteration is the minimum execution unit. For each execution, it consists of multiple kinds of
// Rounds, only after all the rounds are finished, this iteration is considered as completed.
class Iteration {
 public:
  static Iteration &GetInstance() {
    static Iteration instance;
    return instance;
  }

  // Initialize all the rounds in the iteration.
  void InitIteration(const std::shared_ptr<ServerNode> &server_node, const std::vector<RoundConfig> &round_configs,
                     const std::vector<std::shared_ptr<CommunicatorBase>> &communicators);

  // Notify move_to_next_thread_ to move to next iteration.
  void FinishIteration(bool is_last_iter_valid, const std::string &reason);

  // Set current iteration state to running and trigger the event.
  void SetIterationRunning();

  // After hyper-parameters are updated, some rounds and kernels should be reinitialized.
  bool ReInitForUpdatingHyperParams(const std::vector<RoundConfig> &updated_rounds_config);

  void set_loss(float loss);
  void set_accuracy(float accuracy);
  void set_unsupervised_eval(float unsupervised_eval);

  // Need to wait all the rounds to finish before proceed to next iteration.
  void WaitAllRoundsFinish() const;

  // Initialize global iteration timer.
  void InitGlobalIterTimer();

  // Create a thread to record date rate
  void StartThreadToRecordDataRate();

  void SaveModel();
  void SummaryOnIterationFinish(const std::function<void()> &iteration_end_callback);
  void Reset();
  void StartNewInstance();

  void OnRoundLaunchStart();
  void OnRoundLaunchEnd();

  void Stop();

 private:
  Iteration() {}
  ~Iteration();
  Iteration(const Iteration &) = delete;
  Iteration &operator=(const Iteration &) = delete;

  // Reinitialize rounds and round kernels.
  bool ReInitRounds();

  // Summarize metrics for the completed iteration, including iteration time cost, accuracy, loss, etc.
  bool SummarizeIteration();
  void SubmitSummary();
  void GetAllSummaries();
  void SummarizeUnsupervisedEval();
  void InitEventTxtFile();
  void LogFailureEvent(const std::string &node_role, const std::string &node_address, const std::string &event);
  void InitConfig();

  std::shared_ptr<ServerNode> server_node_ = nullptr;

  // All the rounds in the server.
  std::vector<std::shared_ptr<Round>> rounds_;

  // Iteration start time
  fl::Time start_time_;

  // Iteration complete time
  fl::Time complete_time_;

  // The training loss after this federated learning iteration, passed by worker.
  float loss_ = 0.0f;

  // The evaluation result after this federated learning iteration, passed by worker.
  float accuracy_ = 0.0f;

  // The round kernels whose Launch method has not returned yet.
  std::atomic_uint32_t running_round_num_ = 0;

  // for example: "startFLJobTotalClientNum" -> startFLJob total client num
  std::map<std::string, size_t> round_client_num_map_;

  // The number of iteration continuous failure
  uint32_t iteration_fail_num_ = 0;

  // The thread to record data rate
  std::thread data_rate_thread_;

  // The state of data rate thread
  std::atomic_bool is_date_rate_thread_running_ = true;

  // The instance name, just used for metrics
  std::string instance_name_;

  std::string data_rate_file_path_;
  std::string event_file_path_;
  // Whether last iteration is successfully finished and the reason.
  bool is_iteration_valid_;

  // The unsupervised evaluation result after this federated learning iteration.
  float unsupervised_eval_ = 0.0f;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_ITERATION_H_
