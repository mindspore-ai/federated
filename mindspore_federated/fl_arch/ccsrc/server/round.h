/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_ROUND_H_
#define MINDSPORE_CCSRC_FL_SERVER_ROUND_H_

#include <memory>
#include <string>
#include <map>
#include <utility>
#include <vector>
#include "communicator/communicator_base.h"
#include "common/common.h"
#include "server/kernel/round/round_kernel.h"

namespace mindspore {
namespace fl {
namespace server {
// Round helps server to handle network round messages and launch round kernels. One iteration in server consists of
// multiple rounds like startFLJob, updateModel, Push, Pull, etc. Some round kernels may be stateful because of counting
// and timing. So Round helps register counter and timer so that the round kernels only need to focus on the logic.
class Round {
 public:
  Round(const std::string &name, bool check_timeout, uint64_t time_window, bool check_count, uint64_t threshold_count,
        bool per_server_count);
  ~Round() = default;

  void RegisterMsgCallBack(const std::shared_ptr<CommunicatorBase> &communicator);
  void Initialize();

  // After hyper-parameters are updated, some rounds and kernels should be reinitialized.
  bool ReInitForUpdatingHyperParams(uint64_t updated_threshold_count, uint64_t updated_time_window);

  // Bind a round kernel to this Round. This method should be called after Initialize.
  void BindRoundKernel(const std::shared_ptr<kernel::RoundKernel> &kernel);

  // This method is the callback which will be set to the communicator and called after the corresponding round message
  // is sent to the server.
  void LaunchRoundKernel(const std::shared_ptr<MessageHandler> &message);

  // Round needs to be reset after each iteration is finished or its timer expires.
  void Reset();

  void KernelSummarize();

  const std::string &name() const;

  size_t threshold_count() const;

  bool check_timeout() const;

  size_t time_window() const;

  size_t kernel_total_client_num() const;

  size_t kernel_accept_client_num() const;

  size_t kernel_reject_client_num() const;

  void InitKernelClientVisitedNum();

  void InitKernelClientUploadLoss();

  float kernel_upload_loss() const;

  void InitKernelClientUploadAccuracy();

  void InitKernelEvalDataSize();

  void InitKernelTrainDataSize();

  float kernel_upload_accuracy() const;

  size_t kernel_eval_data_size() const;

  std::multimap<uint64_t, size_t> GetSendData() const;

  std::multimap<uint64_t, size_t> GetReceiveData() const;

  std::vector<std::pair<uint64_t, uint32_t>> GetUpdateModelCompleteInfo() const;

  void ResetParticipationTimeAndNum();

  void ClearData();

 private:
  // The callbacks which will be set to DistributedCounterService.
  void OnFirstCountEvent();
  void OnLastCountEvent();

  // Judge whether the training service is available.
  bool IsServerAvailable(std::string *reason);

  RoundConfig config_;
  std::string name_;

  // Whether this round needs to use timer. Most rounds in federated learning with mobile devices scenario need to set
  // check_timeout_ to true.
  bool check_timeout_;

  // The time window duration for this round when check_timeout_ is set to true.
  size_t time_window_;

  // If check_count_ is true, it means the round has to do counting for every round message and the first/last count
  // event will be triggered.
  bool check_count_;

  // The threshold count for this round when check_count_ is set to true. The logic of this round has to check whether
  // the round message count has reached threshold_count_.
  size_t threshold_count_;

  // For update model, local data exists on each server, and we use the total count of all valid servers as the final
  // count.
  bool per_server_count_ = false;

  // The round kernel for this Round.
  std::shared_ptr<kernel::RoundKernel> kernel_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_ROUND_H_
