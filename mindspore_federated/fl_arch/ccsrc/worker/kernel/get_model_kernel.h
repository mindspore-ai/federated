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
#include "worker/fl_worker.h"
#include "python/fl_context.h"

namespace mindspore {
namespace fl {
namespace worker {
namespace kernel {
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
    if (!BuildGetModelReq(fbb_)) {
      MS_LOG(EXCEPTION) << "Building request for FusedPushWeight failed.";
      return dict_data;
    }

    const schema::ResponseGetModel *get_model_rsp = nullptr;
    std::shared_ptr<std::vector<unsigned char>> get_model_rsp_msg = nullptr;
    int response_code = schema::ResponseCode_SucNotReady;
    while (response_code == schema::ResponseCode_SucNotReady) {
      if (!fl::worker::FLWorker::GetInstance().SendToServer(target_server_rank_, fbb_->GetBufferPointer(),
                                                            fbb_->GetSize(), fl::core::TcpUserCommand::kGetModel,
                                                            &get_model_rsp_msg)) {
        MS_LOG(EXCEPTION) << "Sending request for GetModel to server " << target_server_rank_ << " failed.";
        return dict_data;
      }
      flatbuffers::Verifier verifier(get_model_rsp_msg->data(), get_model_rsp_msg->size());
      if (!verifier.VerifyBuffer<schema::ResponseGetModel>()) {
        MS_LOG(EXCEPTION) << "The schema of ResponseGetModel is invalid.";
        return dict_data;
      }

      get_model_rsp = flatbuffers::GetRoot<schema::ResponseGetModel>(get_model_rsp_msg->data());
      MS_EXCEPTION_IF_NULL(get_model_rsp);
      response_code = get_model_rsp->retcode();
      if (response_code == schema::ResponseCode_SUCCEED) {
        break;
      } else if (response_code == schema::ResponseCode_SucNotReady) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        continue;
      } else {
        MS_LOG(EXCEPTION) << "Launching get model for worker failed. Reason: " << get_model_rsp->reason();
      }
    }

    auto feature_map = get_model_rsp->feature_map();
    MS_EXCEPTION_IF_NULL(feature_map);
    if (feature_map->size() == 0) {
      MS_LOG(EXCEPTION) << "Feature map after GetModel is empty.";
      return dict_data;
    }
    auto &feature_maps = FLContext::instance()->feature_maps();
    for (size_t i = 0; i < feature_map->size(); i++) {
      const auto &feature_fbs = feature_map->Get(i);
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

    return dict_data;
  }

  void Init() {
    MS_LOG(INFO) << "Initializing GetModel kernel";
    fbb_ = std::make_shared<FBBuilder>();
    MS_EXCEPTION_IF_NULL(fbb_);

    server_num_ = fl::worker::FLWorker::GetInstance().server_num();
    rank_id_ = fl::worker::FLWorker::GetInstance().rank_id();
    if (rank_id_ == UINT32_MAX) {
      MS_LOG(EXCEPTION) << "Federated worker is not initialized yet.";
      return;
    }
    target_server_rank_ = rank_id_ % server_num_;
    fl_name_ = fl::worker::FLWorker::GetInstance().fl_name();
    MS_LOG(INFO) << "Initializing GetModel kernel. fl_name: " << fl_name_ << ". Request will be sent to server "
                 << target_server_rank_;
  }

 private:
  bool BuildGetModelReq(const std::shared_ptr<FBBuilder> &fbb) {
    MS_EXCEPTION_IF_NULL(fbb_);
    auto fbs_fl_name = fbb->CreateString(fl_name_);
    auto time = fl::core::CommUtil::GetNowTime();
    MS_LOG(INFO) << "now time: " << time.time_str_mill;
    auto fbs_timestamp = fbb->CreateString(std::to_string(time.time_stamp));
    schema::RequestGetModelBuilder req_get_model_builder(*(fbb.get()));
    req_get_model_builder.add_fl_name(fbs_fl_name);
    req_get_model_builder.add_timestamp(fbs_timestamp);
    iteration_ = fl::worker::FLWorker::GetInstance().fl_iteration_num();
    MS_LOG(INFO) << "Get model iteration: " << iteration_;
    req_get_model_builder.add_iteration(SizeToInt(iteration_));
    auto req_get_model = req_get_model_builder.Finish();
    fbb->Finish(req_get_model);
    return true;
  }

  std::shared_ptr<FBBuilder> fbb_;
  uint32_t rank_id_;
  uint32_t server_num_;
  uint32_t target_server_rank_;
  std::string fl_name_;
  uint64_t iteration_;
};
}  // namespace kernel
}  // namespace worker
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_GET_MODEL_H_
