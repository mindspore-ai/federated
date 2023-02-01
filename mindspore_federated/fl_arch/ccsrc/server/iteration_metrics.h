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

#ifndef MINDSPORE_CCSRC_FL_SERVER_ITERATION_METRICS_H_
#define MINDSPORE_CCSRC_FL_SERVER_ITERATION_METRICS_H_

#include <map>
#include <string>
#include <memory>
#include <fstream>
#include "common/fl_context.h"
#include "server/local_meta_store.h"
#include "server/iteration.h"
#include "common/core/comm_util.h"

namespace mindspore {
namespace fl {
namespace server {
constexpr auto kInstanceName = "instanceName";
constexpr auto kFLName = "flName";
constexpr auto kInstanceStatus = "instanceStatus";
constexpr auto kFLIterationNum = "flIterationNum";
constexpr auto kCurIteration = "currentIteration";
constexpr auto kMetricsAuc = "metricsAuc";
constexpr auto kMetricsLoss = "metricsLoss";
constexpr auto kIterExecutionTime = "iterationExecutionTime";
constexpr auto kMetrics = "metrics";
constexpr auto kClientVisitedInfo = "clientVisitedInfo";
constexpr auto kIterationResult = "iterationResult";
constexpr auto kStartTime = "startTime";
constexpr auto kEndTime = "endTime";
constexpr auto kDataRate = "dataRate";
constexpr auto kFailureEvent = "failureEvent";
constexpr auto kMetricsUnsupervisedEval = "unsupervisedEval";

const std::map<cache::InstanceState, std::string> kInstanceStateName = {
  {cache::InstanceState::kStateRunning, "running"},
  {cache::InstanceState::kStateDisable, "disable"},
  {cache::InstanceState::kStateFinish, "finish"}};

class IterationMetrics {
 public:
  IterationMetrics() = default;
  ~IterationMetrics() = default;

  bool Initialize();

  // Gather the details of this iteration and output to the persistent storage.
  bool Summarize();

  // Clear data in persistent storage.
  bool Clear();

  // Setters for the metrics data.
  void set_fl_name(const std::string &fl_name);
  void set_fl_iteration_num(size_t fl_iteration_num);
  void set_cur_iteration_num(size_t cur_iteration_num);
  void set_instance_state(cache::InstanceState state);
  void set_loss(float loss);
  void set_accuracy(float acc);
  void set_iteration_time_cost(uint64_t iteration_time_cost);
  void set_round_client_num_map(const std::map<std::string, size_t> round_client_num_map);
  void set_iteration_result(bool iteration_result);
  void SetStartTime(const Time &start_time);
  void SetEndTime(const Time &end_time);
  void SetInstanceName(const std::string &instance_name);
  void set_unsupervised_eval(const float &unsupervised_eval);

 private:
  // The metrics file object.
  std::fstream metrics_file_;

  // The metrics file path.
  std::string metrics_file_path_;

  // Json object of metrics data.
  nlohmann::basic_json<std::map, std::vector, std::string, bool, int64_t, uint64_t, float> js_;

  // The federated learning job name. Set by fl_context.
  std::string fl_name_;

  // Federated learning iteration number. Set by fl_context.
  // If this number of iterations are completed, one instance is finished.
  size_t fl_iteration_num_ = 0;

  // Current iteration number.
  size_t cur_iteration_num_ = 0;

  // Current instance state.
  cache::InstanceState instance_state_ = cache::InstanceState::kStateFinish;

  // The training loss after this federated learning iteration, passed by worker.
  float loss_ = 0.0f;

  // The evaluation result after this federated learning iteration, passed by worker.
  float accuracy_ = 0.0f;

  // for example: "startFLJobTotalClientNum" -> startFLJob total client num
  std::map<std::string, size_t> round_client_num_map_;

  // The time cost in millisecond for this completed iteration.
  uint64_t iteration_time_cost_ = 0;

  // Current iteration running result.
  bool iteration_result_ = true;

  Time start_time_;
  Time end_time_;

  std::string instance_name_;

  // The unsupervised evaluation result after this federated learning iteration.
  float unsupervised_eval_ = 0.0f;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_ITERATION_METRICS_H_
