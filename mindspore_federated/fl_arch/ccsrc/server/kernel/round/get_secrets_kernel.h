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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_GET_SECRETS_KERNEL_H
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_GET_SECRETS_KERNEL_H

#include <vector>
#include <memory>
#include "common/common.h"
#include "server/kernel/round/round_kernel.h"
#include "server/kernel/round/round_kernel_factory.h"
#include "armour/cipher/cipher_shares.h"
#include "server/executor.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {

class GetSecretsKernel : public RoundKernel {
 public:
  GetSecretsKernel() = default;
  ~GetSecretsKernel() override = default;
  void InitKernel(size_t required_cnt) override;
  bool Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) override;

 private:
  armour::CipherShares *cipher_share_ = nullptr;
  sigVerifyResult VerifySignature(const schema::GetShareSecrets *get_secrets_req);
  bool CountForGetSecrets(const std::shared_ptr<FBBuilder> &fbb, const schema::GetShareSecrets *get_secrets_req,
                          const size_t iter_num);
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_GET_SECRETS_KERNEL_H
