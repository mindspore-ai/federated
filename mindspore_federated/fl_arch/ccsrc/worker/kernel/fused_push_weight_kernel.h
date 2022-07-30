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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PS_FUSED_PUSH_WEIGHT_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PS_FUSED_PUSH_WEIGHT_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <utility>
#include <map>
#include "worker/kernel/abstract_kernel.h"
#include "common/fl_context.h"
#include "worker/hybrid_worker.h"

namespace mindspore {
namespace fl {
namespace worker {
namespace kernel {
// The duration between two PushWeight requests when return code is ResponseCode_SucNotReady.
constexpr int kRetryDurationOfPushWeights = 200;
class FusedPushWeightKernelMod : public AbstractKernel {
 public:
  FusedPushWeightKernelMod() : fl_iteration_(0) {}
  ~FusedPushWeightKernelMod() override = default;

  static std::shared_ptr<FusedPushWeightKernelMod> GetInstance() {
    static std::shared_ptr<FusedPushWeightKernelMod> instance = nullptr;
    if (instance == nullptr) {
      instance.reset(new FusedPushWeightKernelMod());
    }
    return instance;
  }

  bool Launch(const std::map<std::string, std::vector<float>> &weight_datas) {
    MS_LOG(INFO) << "Launch FusedPushWeightKernelMod.";
    FBBuilder fbb;

    fl_iteration_++;
    MS_LOG(INFO) << "Launching pushing weight for federated learning iteration " << fl_iteration_;
    if (!BuildPushWeightReq(&fbb, weight_datas)) {
      MS_LOG(EXCEPTION) << "Building request for FusedPushWeight failed.";
    }

    // The server number may change after scaling in/out.
    std::shared_ptr<std::vector<unsigned char>> push_weight_rsp_msg = nullptr;
    const schema::ResponsePushWeight *push_weight_rsp = nullptr;
    int retcode = schema::ResponseCode_SucNotReady;
    while (retcode == schema::ResponseCode_SucNotReady) {
      if (ExitHandler::Instance().HasStopped()) {
        MS_LOG(WARNING) << "Worker has finished.";
        return true;
      }
      if (!fl::worker::HybridWorker::GetInstance().SendToServer(
            fbb.GetBufferPointer(), fbb.GetSize(), fl::TcpUserCommand::kPushWeight, &push_weight_rsp_msg)) {
        MS_LOG(WARNING) << "Sending request for FusedPushWeight to server failed.";
        retcode = schema::ResponseCode_SucNotReady;
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDurationOfPushWeights));
        continue;
      }
      MS_EXCEPTION_IF_NULL(push_weight_rsp_msg);

      push_weight_rsp = flatbuffers::GetRoot<schema::ResponsePushWeight>(push_weight_rsp_msg->data());
      MS_EXCEPTION_IF_NULL(push_weight_rsp);
      retcode = push_weight_rsp->retcode();
      if (retcode == schema::ResponseCode_SucNotReady) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDurationOfPushWeights));
        fl_iteration_ = push_weight_rsp->iteration();
        MS_LOG(DEBUG) << "Server is not ready for pushing weight yet. Reason: " << push_weight_rsp->reason()->str()
                      << ". Retry later.";
        if (!BuildPushWeightReq(&fbb, weight_datas)) {
          MS_LOG(EXCEPTION) << "Building request for FusedPushWeight failed.";
        }
        continue;
      } else if (retcode != schema::ResponseCode_SUCCEED) {
        MS_LOG(WARNING) << "FusedPushWeight failed. Server return code: " << push_weight_rsp->retcode()
                        << ", reason: " << push_weight_rsp->reason()->str();
      } else {
        MS_LOG(DEBUG) << "FusedPushWeight succeed.";
      }
    }

    MS_LOG(INFO) << "Push weights for iteration: " << fl_iteration_ << " success.";
    return true;
  }

  void Init() override {}

 private:
  bool BuildPushWeightReq(FBBuilder *fbb, const std::map<std::string, std::vector<float>> &weight_datas) {
    if (fbb == nullptr) {
      return false;
    }
    fbb->Clear();
    std::vector<flatbuffers::Offset<schema::FeatureMap>> fbs_feature_maps;
    for (auto &weight : weight_datas) {
      const std::string &weight_name = weight.first;
      auto &weight_data = weight.second;
      auto fbs_weight_fullname = fbb->CreateString(weight_name);
      auto fbs_weight_data = fbb->CreateVector(weight_data);
      auto fbs_feature_map = schema::CreateFeatureMap(*fbb, fbs_weight_fullname, fbs_weight_data);
      fbs_feature_maps.push_back(fbs_feature_map);
    }
    auto fbs_feature_maps_vector = fbb->CreateVector(fbs_feature_maps);

    schema::RequestPushWeightBuilder req_push_weight_builder(*fbb);
    req_push_weight_builder.add_iteration(fl_iteration_);
    req_push_weight_builder.add_feature_map(fbs_feature_maps_vector);
    auto req_push_weight = req_push_weight_builder.Finish();
    fbb->Finish(req_push_weight);
    return true;
  }

  size_t fl_iteration_;
};
}  // namespace kernel
}  // namespace worker
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PS_FUSED_PUSH_WEIGHT_KERNEL_H_
