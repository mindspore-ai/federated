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

#ifndef MINDSPORE_CCSRC_FL_SERVER_EXECUTOR_H_
#define MINDSPORE_CCSRC_FL_SERVER_EXECUTOR_H_

#include <map>
#include <set>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include "armour/cipher/cipher_unmask.h"
#include "common/common.h"
#include "server/model_store.h"
#include "server/server_node.h"
#include "common/constants.h"

namespace mindspore {
namespace fl {
namespace server {
struct ParamAggregationInfo {
  std::string name;
  uint8_t *weight_data = nullptr;
  size_t weight_size = 0;  // bytes len of weight_data
  size_t data_size = 0;    // batch size
  bool *require_aggr;
};
// Executor is the entrance for server to handle aggregation, optimizing, model querying, etc. It handles
// logics relevant to kernel launching.
class Executor {
 public:
  static Executor &GetInstance() {
    static Executor instance;
    return instance;
  }

  void Initialize(const std::vector<InputWeight> &feature_map, const std::shared_ptr<ServerNode> &server_node);

  // Returns whether the executor singleton is already initialized.
  bool initialized() const;

  FlStatus SyncLatestModelFromOtherServers();

  FlStatus CheckUpdatedModel(const std::map<std::string, Address> &feature_map, const std::string &update_model_fl_id);
  // Called in federated learning training mode. Update value for parameters.
  void HandleModelUpdate(const std::map<std::string, Address> &feature_map, size_t data_size);

  std::map<std::string, Address> ParseFeatureMap(const schema::RequestPushWeight *push_weight_req);
  FlStatus HandlePullWeightRequest(const uint8_t *req_data, size_t len, FBBuilder *fbb);

  bool OnReceiveModelWeight(const uint8_t *proto_model_data, size_t len);

  void RunWeightAggregation();
  // Reset the aggregation status for all aggregation kernels in the server.
  bool ResetAggregationStatus();

  bool IsAggregationSkip() const;

  // Judge whether aggregation processes for all weights/gradients are completed.
  bool IsAggregationDone() const;

  void SetIterationModelFinished();
  bool IsIterationModelFinished(uint64_t iteration_num) const;

  // whether the unmasking is completed.
  bool IsUnmasked() const;
  void TodoUnmask();
  void OnPushMetrics();

  // Returns whole model in key-value where key refers to the parameter name.
  ModelItemPtr GetModel();

  ModelItemPtr GetModelByIteration(uint64_t iteration_num);
  bool GetModelByIteration(uint64_t iteration_num, ProtoModel *proto_model);

  void FinishIteration(bool is_last_iter_valid, const std::string &in_reason);

  bool TransModel2ProtoModel(uint64_t iteration_num, const ModelItemPtr &model, ProtoModel *proto_model);

  // Forcibly overwrite specific weights in overwriteWeights message.
  bool HandlePushWeight(const std::map<std::string, Address> &feature_map);

 private:
  Executor() = default;
  ~Executor() = default;
  Executor(const Executor &) = delete;
  Executor &operator=(const Executor &) = delete;

  bool GetServersForAllReduce(std::map<std::string, std::string> *all_reduce_server_map);
  void BroadcastModelWeight(const std::map<std::string, std::string> &broadcast_src_server_map);
  FlStatus BuildPullWeightRsp(size_t iteration, const std::vector<std::string> &param_names, FBBuilder *fbb);

  void SetSkipAggregation();
  bool RunWeightAggregationInner(const std::map<std::string, std::string> &server_map);
  // The unmasking method for pairwise encrypt algorithm.
  void Unmask();

  std::mutex parameter_mutex_;
  ModelItemPtr model_aggregation_ = nullptr;
  std::map<std::string, ParamAggregationInfo> param_aggregation_info_;
  // whether model in model_aggregation_ has finished
  bool model_finished_ = false;

  bool initialized_ = false;

  armour::CipherUnmask cipher_unmask_;

  // The flag refers to the unmasking status
  std::atomic<bool> unmasked_ = false;
  uint64_t finish_model_iteration_num_ = 0;
  bool is_aggregation_done_ = false;
  bool is_aggregation_skip_ = false;

  std::shared_ptr<ServerNode> server_node_ = nullptr;

  bool can_unmask_ = false;
  // servers participating in gradient aggregation
  std::map<std::string, std::string> all_reduce_server_map_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_EXECUTOR_H_
