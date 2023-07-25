/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "server/kernel/round/get_result_kernel.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void GetResultKernel::InitKernel(size_t) { InitClientVisitedNum(); }

bool GetResultKernel::Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) {
  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, reason.c_str(), reason.size());
    return true;
  }

  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::RequestGetResult>()) {
    std::string reason = "The schema of RequestGetResult is invalid.";
    BuildGetResultRsp(fbb, schema::ResponseCode_RequestError, reason,
                      cache::InstanceContext::Instance().iteration_num(), "");
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  retry_count_ += 1;
  if (retry_count_.load() % kPrintGetResultForEveryRetryTime == 1) {
    MS_LOG(DEBUG) << "Launching GetResultKernel kernel. Retry count is " << retry_count_.load();
  }

  const schema::RequestGetResult *get_result_req = flatbuffers::GetRoot<schema::RequestGetResult>(req_data);
  if (get_result_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for RequestGetResult.";
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, reason.c_str(), reason.size());
    return true;
  }
  GetResult(get_result_req, message);
  return true;
}

bool GetResultKernel::Reset() {
  MS_LOG(INFO) << "Get result kernel reset!";
  retry_count_ = 0;
  return true;
}

void GetResultKernel::GetResult(const schema::RequestGetResult *get_result_req,
                                const std::shared_ptr<MessageHandler> &message) {
  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr) {
    std::string reason = "FBBuilder builder is nullptr.";
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, reason.c_str(), reason.size());
    return;
  }
  auto next_req_time = LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp);
  size_t current_iter = cache::InstanceContext::Instance().iteration_num();
  size_t get_result_iter = IntToSize(get_result_req->iteration());
  const auto &iter_to_model = ModelStore::GetInstance().iteration_to_model();
  size_t model_latest_iter_num = iter_to_model.rbegin()->first;
  // If this iteration is not finished yet, return ResponseCode_SucNotReady so that clients could get result later.
  if (current_iter == get_result_iter && model_latest_iter_num != current_iter) {
    std::string reason = "The model is not ready yet for iteration " + std::to_string(get_result_iter) +
                         ". Maybe this is because\n" + "1. Client doesn't not send enough update model request.\n" +
                         "2. Worker has not push weights to server.";
    BuildGetResultRsp(fbb, schema::ResponseCode_SucNotReady, reason, current_iter, std::to_string(next_req_time));
    if (retry_count_.load() % kPrintGetResultForEveryRetryTime == 1) {
      MS_LOG(DEBUG) << reason;
    }
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return;
  }

  IncreaseAcceptClientNum();
  BuildGetResultRsp(fbb, schema::ResponseCode_SUCCEED, "Get result for iteration " + std::to_string(get_result_iter),
                    current_iter, std::to_string(next_req_time));
  SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
  return;
}

void GetResultKernel::BuildGetResultRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                        const std::string &reason, const size_t iter, const std::string &timestamp) {
  if (fbb == nullptr) {
    MS_LOG(ERROR) << "Input fbb is nullptr.";
    return;
  }
  auto fbs_reason = fbb->CreateString(reason);
  auto fbs_timestamp = fbb->CreateString(timestamp);

  schema::ResponseGetResultBuilder rsp_get_result_builder(*(fbb.get()));
  rsp_get_result_builder.add_retcode(static_cast<int>(retcode));
  rsp_get_result_builder.add_reason(fbs_reason);
  rsp_get_result_builder.add_iteration(static_cast<int>(iter));
  rsp_get_result_builder.add_timestamp(fbs_timestamp);
  auto rsp_get_result = rsp_get_result_builder.Finish();
  fbb->Finish(rsp_get_result);
  return;
}

REG_ROUND_KERNEL(getResult, GetResultKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
