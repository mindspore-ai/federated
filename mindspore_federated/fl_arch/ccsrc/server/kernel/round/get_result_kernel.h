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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_GET_RESULT_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_GET_RESULT_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/common.h"
#include "server/executor.h"
#include "server/kernel/round/round_kernel.h"
#include "server/kernel/round/round_kernel_factory.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
constexpr uint32_t kPrintGetResultForEveryRetryTime = 50;
class GetResultKernel : public RoundKernel {
 public:
  GetResultKernel() = default;
  ~GetResultKernel() override = default;

  void InitKernel(size_t) override;
  bool Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) override;
  bool Reset() override;

 private:
  void GetResult(const schema::RequestGetResult *get_result_req, const std::shared_ptr<MessageHandler> &message);
  void BuildGetResultRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                         const std::string &reason, const size_t iter, const std::string &timestamp);

  // The count of retrying because the iteration is not finished.
  std::atomic<uint64_t> retry_count_ = 0;
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_GET_RESULT_KERNEL_H_
