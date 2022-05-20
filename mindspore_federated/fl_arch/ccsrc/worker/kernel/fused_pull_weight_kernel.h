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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PS_FUSED_PULL_WEIGHT_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PS_FUSED_PULL_WEIGHT_KERNEL_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <utility>
#include "worker/kernel/abstract_kernel.h"
#include "python/fl_context.h"
#include "schema/fl_job_generated.h"
#include "worker/fl_worker.h"

namespace mindspore {
namespace fl {
namespace worker {
namespace kernel {
// The duration between two PullWeight requests when return code is ResponseCode_SucNotReady.
constexpr int kRetryDurationOfPullWeights = 200;
class FusedPullWeightKernelMod : public AbstractKernel {
 public:
  FusedPullWeightKernelMod()
      : server_num_(0), indices_({}), weight_full_names_({}), fl_iteration_(0), total_iteration_(0) {}
  ~FusedPullWeightKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs) {
    MS_LOG(DEBUG) << "Launch FusedPullWeightKernelMod.";
    if (inputs.size() != weight_full_names_.size()) {
      MS_LOG(EXCEPTION) << "Input number is " << inputs.size() << ", but FusedPullWeightKernelMod needs "
                        << weight_full_names_.size() << " weights as inputs.";
    }

    total_iteration_++;
    uint64_t step_num_per_iteration = fl::worker::FLWorker::GetInstance().worker_step_num_per_iteration();
    if (step_num_per_iteration == 0) {
      MS_LOG(EXCEPTION) << "step number per iteration should not be 0";
    }
    MS_LOG(INFO) << "Try to pull weights. Local step number: " << total_iteration_
                 << ", step number needs to run per iteration: " << step_num_per_iteration;
    // The worker has to train kWorkerTrainStepNum standalone iterations before it communicates with server.
    if (step_num_per_iteration != fl::kOneStepPerIteration &&
        total_iteration_ % step_num_per_iteration != fl::kTrainBeginStepNum) {
      return true;
    }

    fl_iteration_++;
    MS_LOG(INFO) << "Launching pulling weight for federated learning iteration " << fl_iteration_;

    std::shared_ptr<FBBuilder> fbb;
    std::shared_ptr<std::vector<unsigned char>> pull_weight_rsp_msg = nullptr;
    const schema::ResponsePullWeight *pull_weight_rsp = nullptr;
    int retcode = schema::ResponseCode_SucNotReady;
    while (retcode == schema::ResponseCode_SucNotReady) {
      if (!fl::worker::FLWorker::GetInstance().running()) {
        MS_LOG(WARNING) << "Worker has finished.";
        return true;
      }
      // Recreate fbb to avoid memory leak of FlatBuffers.
      fbb = std::make_shared<FBBuilder>();
      if (!BuildPullWeightReq(fbb)) {
        MS_LOG(ERROR) << "Building request for FusedPullWeight failed.";
        continue;
      }

      if (!fl::worker::FLWorker::GetInstance().SendToServer(
            0, fbb->GetBufferPointer(), fbb->GetSize(), fl::core::TcpUserCommand::kPullWeight, &pull_weight_rsp_msg)) {
        MS_LOG(WARNING) << "Sending request for FusedPullWeight to server 0 failed. Retry later.";
        retcode = schema::ResponseCode_SucNotReady;
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDurationOfPullWeights));
        continue;
      }

      if (pull_weight_rsp_msg == nullptr || pull_weight_rsp_msg->data() == nullptr) {
        continue;
      }
      auto pull_weight_rsp_data = reinterpret_cast<const uint8_t *>(pull_weight_rsp_msg->data());

      flatbuffers::Verifier verifier(pull_weight_rsp_data, sizeof(unsigned char) * pull_weight_rsp_msg->size());
      if (!verifier.VerifyBuffer<schema::ResponsePullWeight>()) {
        MS_LOG(ERROR) << "The schema of ResponsePullWeight is invalid.";
        continue;
      }

      pull_weight_rsp = flatbuffers::GetRoot<schema::ResponsePullWeight>(pull_weight_rsp_msg->data());
      if (pull_weight_rsp == nullptr) {
        continue;
      }

      retcode = pull_weight_rsp->retcode();
      if (retcode == schema::ResponseCode_SucNotReady) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDurationOfPullWeights));
        fl_iteration_ = pull_weight_rsp->iteration();
        MS_LOG(DEBUG) << "Server is not ready for downloading yet. Reason: " << pull_weight_rsp->reason()->str()
                      << ". Retry later.";
        continue;
      } else if (retcode != schema::ResponseCode_SUCCEED) {
        MS_LOG(WARNING) << "FusedPullWeight failed. Server return code: " << pull_weight_rsp->retcode()
                        << ", reason: " << pull_weight_rsp->reason()->str();
      } else {
        MS_LOG(DEBUG) << "FusedPullWeight succeed.";
      }
    }

    auto feature_map = ParseFeatureMap(pull_weight_rsp);
    for (size_t i = 0; i < weight_full_names_.size(); i++) {
      const std::string &weight_name = weight_full_names_[i];
      if (feature_map.count(weight_name) == 0) {
        MS_LOG(EXCEPTION) << "The weights for " << weight_name << " is not pulled from server.";
      }
      int ret =
        memcpy_s(inputs[i]->addr, inputs[i]->size, feature_map[weight_name].addr, feature_map[weight_name].size);
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
      }
    }
    MS_LOG(INFO) << "Pull weights for " << weight_full_names_ << " success. Iteration: " << fl_iteration_;
    fl::worker::FLWorker::GetInstance().SetIterationRunning();
    return true;
  }

  void Init() override {}

  void InitKernel() override { return; }

 protected:
  void InitSizeLists() { return; }

 private:
  template <typename T>
  void InitFunc() {}

  using FusedPullWeightInitFunc = std::function<void(FusedPullWeightKernelMod *)>;
  static std::vector<std::pair<KernelAttr, FusedPullWeightInitFunc>> func_list_;
  FusedPullWeightInitFunc init_func_;

  uint32_t server_num_;
  std::vector<int64_t> indices_;
  std::vector<std::string> weight_full_names_;
  uint64_t fl_iteration_;
  uint64_t total_iteration_;
};
}  // namespace kernel
}  // namespace worker
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PS_FUSED_PULL_WEIGHT_KERNEL_H_
