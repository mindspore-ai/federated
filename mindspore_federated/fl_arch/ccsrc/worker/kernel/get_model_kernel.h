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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_GET_MODEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_GET_MODEL_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <functional>
#include "worker/kernel/abstract_kernel.h"
#include "worker/cloud_worker.h"
#include "common/fl_context.h"

namespace mindspore {
namespace fl {
namespace worker {
namespace kernel {
constexpr size_t kRetryTimesOfGetModel = 512;
constexpr size_t kSleepMillisecondsOfGetModel = 1000;
class GetModelKernelMod : public AbstractKernel {
 public:
  GetModelKernelMod() = default;
  ~GetModelKernelMod() override = default;

  static std::shared_ptr<GetModelKernelMod> GetInstance() {
    static std::shared_ptr<GetModelKernelMod> instance = nullptr;
    if (instance == nullptr) {
      instance.reset(new GetModelKernelMod());
      instance->Init();
    }
    return instance;
  }

  py::dict Launch() {
    MS_LOG(INFO) << "Launching client GetModelKernelMod";
    py::dict dict_data;
    FBBuilder fbb;
    if (!BuildGetModelReq(&fbb)) {
      MS_LOG(EXCEPTION) << "Building request for FusedPushWeight failed.";
    }
    size_t retryTimes = 0;
    while (retryTimes < kRetryTimesOfGetModel) {
      auto response_msg =
        fl::worker::CloudWorker::GetInstance().SendToServerSync(fbb.GetBufferPointer(), fbb.GetSize(), kernel_path_);

      if (response_msg == nullptr) {
        MS_LOG(WARNING) << "The response message is invalid.";
        continue;
      }
      retryTimes += 1;
      std::this_thread::sleep_for(std::chrono::milliseconds(kSleepMillisecondsOfGetModel));
      flatbuffers::Verifier verifier(response_msg->data(), response_msg->size());
      if (!verifier.VerifyBuffer<schema::ResponseGetModel>()) {
        MS_LOG(INFO) << "The schema of response message is invalid.";
        continue;
      }
      const schema::ResponseGetModel *get_model_rsp =
        flatbuffers::GetRoot<schema::ResponseGetModel>(response_msg->data());
      MS_ERROR_IF_NULL_W_RET_VAL(get_model_rsp, dict_data);
      auto response_code = get_model_rsp->retcode();
      if (response_code == schema::ResponseCode_SUCCEED) {
        MS_LOG(INFO) << "Get model response code from server is success.";
      } else if (response_code == schema::ResponseCode_SucNotReady) {
        MS_LOG(INFO) << "Get model response code from server is not ready.";
        continue;
      } else {
        MS_LOG(EXCEPTION) << "Launching get model for worker failed. Reason: " << get_model_rsp->reason();
      }
      auto feature_map = get_model_rsp->feature_map();
      MS_EXCEPTION_IF_NULL(feature_map);

      if (feature_map->size() == 0) {
        MS_LOG(WARNING) << "Feature map after GetModel is empty.";
        continue;
      }
      for (size_t i = 0; i < feature_map->size(); i++) {
        const auto &feature_fbs = feature_map->Get(i);
        const auto &feature_data_fbs = feature_fbs->data();

        std::string weight_fullname = feature_fbs->weight_fullname()->str();
        float *weight_data = const_cast<float *>(feature_data_fbs->data());
        std::vector<float> weight_data_vec(weight_data, weight_data + feature_data_fbs->size());
        dict_data[py::str(weight_fullname)] = weight_data_vec;
      }
      MS_LOG(INFO) << "Get model from server successful.";
      break;
    }
    return dict_data;
  }

  void Init() override {
    fl_name_ = fl::worker::CloudWorker::GetInstance().fl_name();
    kernel_path_ = "/getModel";
    MS_LOG(INFO) << "Initializing GetModel kernel. fl_name: " << fl_name_ << ". Request will be sent to server";
  }

 private:
  bool BuildGetModelReq(FBBuilder *fbb) {
    MS_EXCEPTION_IF_NULL(fbb);
    auto fbs_fl_name = fbb->CreateString(fl_name_);
    auto time = fl::CommUtil::GetNowTime();
    auto fbs_timestamp = fbb->CreateString(std::to_string(time.time_stamp));
    schema::RequestGetModelBuilder req_get_model_builder(*fbb);
    req_get_model_builder.add_fl_name(fbs_fl_name);
    req_get_model_builder.add_timestamp(fbs_timestamp);
    iteration_ = fl::worker::CloudWorker::GetInstance().fl_iteration_num();
    MS_LOG(INFO) << "now time: " << time.time_str_mill << ", get model iteration: " << iteration_;
    req_get_model_builder.add_iteration(SizeToInt(iteration_));
    auto req_get_model = req_get_model_builder.Finish();
    fbb->Finish(req_get_model);
    return true;
  }
  std::string fl_name_;
  uint64_t iteration_;
  std::string kernel_path_;
};
}  // namespace kernel
}  // namespace worker
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_GET_MODEL_H_
