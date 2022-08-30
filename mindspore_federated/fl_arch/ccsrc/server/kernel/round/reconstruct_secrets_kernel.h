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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_RECONSTRUCT_SECRETS_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_RECONSTRUCT_SECRETS_KERNEL_H_

#include <vector>
#include <memory>
#include <string>
#include "common/common.h"
#include "server/kernel/round/round_kernel.h"
#include "server/kernel/round/round_kernel_factory.h"
#include "armour/cipher/cipher_reconstruct.h"
#include "server/executor.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
class ReconstructSecretsKernel : public RoundKernel {
 public:
  ReconstructSecretsKernel() = default;
  ~ReconstructSecretsKernel() override = default;

  void InitKernel(size_t required_cnt) override;
  bool Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) override;
  void OnLastCountEvent() override;

 private:
  std::string name_unmask_;
  armour::CipherReconStruct cipher_reconstruct_;
  sigVerifyResult VerifySignature(const schema::SendReconstructSecret *reconstruct_secret_req);
  bool checkReachThreshold(const std::vector<std::string> update_model_clients, const int cur_iterator,
                           const std::string next_req_time, std::shared_ptr<FBBuilder> fbb, const std::string fl_id);
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_RECONSTRUCT_SECRETS_KERNEL_H_
