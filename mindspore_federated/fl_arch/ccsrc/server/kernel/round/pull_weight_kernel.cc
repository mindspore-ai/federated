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

#include "server/kernel/round/pull_weight_kernel.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "server/model_store.h"
#include "server/server.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void PullWeightKernel::InitKernel(size_t) {}

bool PullWeightKernel::Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) {
  MS_LOG(DEBUG) << "Launching PullWeightKernel kernel.";
  if (req_data == nullptr || len == 0) {
    std::string reason = "req_data is nullptr or len is 0.";
    MS_LOG(ERROR) << reason;
    return false;
  }
  auto current_iter = cache::InstanceContext::Instance().iteration_num();
  FBBuilder fbb;
  auto status = Executor::GetInstance().HandlePullWeightRequest(req_data, len, &fbb);
  if (status == kAggregationNotDone) {
    // this server has skip aggregation, pull weight from other servers
    if (Executor::GetInstance().IsAggregationSkip()) {
      VectorPtr output = nullptr;
      auto ret = Server::GetInstance().PullWeight(req_data, len, &output);
      if (ret && output != nullptr) {
        MS_LOG(INFO) << "Pulling weight from other servers for iteration " << current_iter << " succeeds.";
        SendResponseMsg(message, output->data(), output->size());
        return true;
      }
    }
  }
  // aggregation not done yet, or some other error happened
  if (!status.IsSuccess()) {
    BuildErrorPullWeightRsp(status, current_iter, &fbb);
    SendResponseMsg(message, fbb.GetBufferPointer(), fbb.GetSize());
    return true;
  }
  MS_LOG(INFO) << "Pulling weight for iteration " << current_iter << " succeeds.";
  SendResponseMsg(message, fbb.GetBufferPointer(), fbb.GetSize());
  return true;
}

bool PullWeightKernel::Reset() {
  retry_count_ = 0;
  return true;
}

void PullWeightKernel::BuildErrorPullWeightRsp(const FlStatus &status, size_t iteration, FBBuilder *fbb) {
  if (fbb == nullptr) {
    MS_LOG(ERROR) << "fbb is nullptr.";
    return;
  }
  auto fbs_reason = fbb->CreateString(status.StatusMessage());
  auto retcode = schema::ResponseCode_RequestError;
  auto status_code = status.GetCode();
  if (status_code == kNotReadyError) {
    retcode = schema::ResponseCode_SucNotReady;
  } else if (status_code == kAggregationNotDone) {
    retcode = schema::ResponseCode_SucNotReady;
    retry_count_ += 1;
    std::string reason = "The aggregation for the weights is not done yet.";
    if (retry_count_.load() % kPrintPullWeightForEveryRetryTime == 1) {
      MS_LOG(WARNING) << reason << " Retry count is " << retry_count_.load();
    }
  }
  std::vector<flatbuffers::Offset<schema::FeatureMap>> fbs_feature_maps;
  auto fbs_feature_maps_vector = fbb->CreateVector(fbs_feature_maps);

  schema::ResponsePullWeightBuilder rsp_pull_weight_builder(*fbb);
  rsp_pull_weight_builder.add_retcode(SizeToInt(retcode));
  rsp_pull_weight_builder.add_reason(fbs_reason);
  rsp_pull_weight_builder.add_iteration(SizeToInt(iteration));
  rsp_pull_weight_builder.add_feature_map(fbs_feature_maps_vector);
  auto rsp_pull_weight = rsp_pull_weight_builder.Finish();
  fbb->Finish(rsp_pull_weight);
}

REG_ROUND_KERNEL(pullWeight, PullWeightKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
