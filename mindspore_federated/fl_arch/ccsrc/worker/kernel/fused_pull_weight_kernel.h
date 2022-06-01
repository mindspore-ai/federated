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
#include "worker/fl_worker.h"

namespace mindspore {
namespace fl {
namespace worker {
namespace kernel {
// The duration between two PullWeight requests when return code is ResponseCode_SucNotReady.
constexpr int kRetryDurationOfPullWeights = 200;
class FusedPullWeightKernelMod : public AbstractKernel {
 public:
  FusedPullWeightKernelMod() : fl_iteration_(0) {}
  ~FusedPullWeightKernelMod() override = default;

  static std::shared_ptr<FusedPullWeightKernelMod> GetInstance() {
    static std::shared_ptr<FusedPullWeightKernelMod> instance = nullptr;
    if (instance == nullptr) {
      instance.reset(new FusedPullWeightKernelMod());
      instance->Init();
    }
    return instance;
  }

  py::dict Launch() {
    MS_LOG(INFO) << "Launch FusedPullWeightKernelMod.";
    py::dict dict_data;
    fl_iteration_++;
    MS_LOG(INFO) << "Launching pulling weight for federated learning iteration " << fl_iteration_;

    std::shared_ptr<FBBuilder> fbb;
    std::shared_ptr<std::vector<unsigned char>> pull_weight_rsp_msg = nullptr;
    const schema::ResponsePullWeight *pull_weight_rsp = nullptr;
    int retcode = schema::ResponseCode_SucNotReady;
    while (retcode == schema::ResponseCode_SucNotReady) {
      if (!fl::worker::FLWorker::GetInstance().running()) {
        MS_LOG(WARNING) << "Worker has finished.";
        return dict_data;
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
      } else if (retcode != schema::ResponseCode_SUCCEED) {
        MS_LOG(WARNING) << "FusedPullWeight failed. Server return code: " << pull_weight_rsp->retcode()
                        << ", reason: " << pull_weight_rsp->reason()->str();
      } else {
        MS_LOG(DEBUG) << "FusedPullWeight succeed.";
      }
    }

    auto feature_map_fbs = pull_weight_rsp->feature_map();
    if (feature_map_fbs->size() == 0) {
      MS_LOG(EXCEPTION) << "Feature map fbs size is empty.";
    }
    auto &feature_maps = FLContext::instance()->feature_maps();
    for (size_t i = 0; i < feature_map_fbs->size(); i++) {
      const auto &feature_fbs = feature_map_fbs->Get(i);
      const auto &feature_data_fbs = feature_fbs->data();

      std::string weight_fullname = feature_fbs->weight_fullname()->str();
      float *weight_data = const_cast<float *>(feature_data_fbs->data());
      std::vector<float> weight_data_vec(weight_data, weight_data + feature_data_fbs->size());

      Feature feature = feature_maps[weight_fullname];
      py::list data_list;
      data_list.append(feature.weight_type);
      data_list.append(feature.weight_shape);
      data_list.append(weight_data_vec);
      data_list.append(weight_data_vec.size());
      dict_data[py::str(weight_fullname)] = data_list;
    }
    MS_LOG(INFO) << "Pull weights for iteration: " << fl_iteration_ << " success.";
    fl::worker::FLWorker::GetInstance().SetIterationRunning();
    return dict_data;
  }

  void Init() {
    auto &feature_maps = FLContext::instance()->feature_maps();
    for (const auto &item : feature_maps) {
      weight_full_names_.push_back(item.first);
    }
    MS_LOG(INFO) << "Pull weight full names: " << weight_full_names_;
  }

 private:
 bool BuildPullWeightReq(std::shared_ptr<fl::FBBuilder> fbb) {
    MS_EXCEPTION_IF_NULL(fbb);
    std::vector<flatbuffers::Offset<flatbuffers::String>> fbs_weight_names;
    for (const std::string &weight_name : weight_full_names_) {
      auto fbs_weight_name = fbb->CreateString(weight_name);
      fbs_weight_names.push_back(fbs_weight_name);
    }
    auto fbs_weight_names_vector = fbb->CreateVector(fbs_weight_names);
    schema::RequestPullWeightBuilder req_pull_weight_builder(*(fbb.get()));
    req_pull_weight_builder.add_weight_names(fbs_weight_names_vector);
    req_pull_weight_builder.add_iteration(fl_iteration_);
    auto req_pull_weight = req_pull_weight_builder.Finish();
    fbb->Finish(req_pull_weight);
    return true;
  }

  uint64_t fl_iteration_;
  std::vector<std::string> weight_full_names_;
};
}  // namespace kernel
}  // namespace worker
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PS_FUSED_PULL_WEIGHT_KERNEL_H_
