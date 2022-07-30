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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_EXCHANGE_KEYS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_EXCHANGE_KEYS_H_

#include <vector>
#include <string>
#include <memory>
#include "worker/cloud_worker.h"
#include "armour/secure_protocol/key_agreement.h"
#include "worker/kernel/abstract_kernel.h"

namespace mindspore {
namespace fl {
namespace worker {
namespace kernel {
class ExchangeKeysKernelMod : public AbstractKernel {
 public:
  ExchangeKeysKernelMod() = default;
  ~ExchangeKeysKernelMod() override = default;

  void Init() override;
  bool Launch();

 private:
  bool BuildExchangeKeysReq(FBBuilder *fbb);
  std::vector<uint8_t> GetPubicKeyBytes();

  std::string fl_id_;
  armour::PrivateKey *secret_prikey_ = nullptr;
  std::string kernel_path_;
};
}  // namespace kernel
}  // namespace worker
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_EXCHANGE_KEYS_H_
