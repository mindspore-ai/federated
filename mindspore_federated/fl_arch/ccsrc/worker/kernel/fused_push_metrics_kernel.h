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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_PUSH_METRICS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_PUSH_METRICS_H_

#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <utility>
#include "worker/kernel/abstract_kernel.h"
#include "worker/hybrid_worker.h"

namespace mindspore {
namespace fl {
namespace worker {
namespace kernel {
// The duration between two PushMetrics requests.
constexpr int kRetryDurationOfPushMetrics = 500;
// Retry for 30 minutes.
constexpr int kMaxRetryTime = 3600;
class FusedPushMetricsKernelMod : public AbstractKernel {
 public:
  FusedPushMetricsKernelMod() : total_iteration_(0) {}
  ~FusedPushMetricsKernelMod() override = default;

  void Init() override {}

  static std::shared_ptr<FusedPushMetricsKernelMod> GetInstance() {
    static std::shared_ptr<FusedPushMetricsKernelMod> instance = nullptr;
    if (instance == nullptr) {
      instance.reset(new FusedPushMetricsKernelMod());
      instance->Init();
    }
    return instance;
  }

  bool Launch(float loss, float accuracy) {
    MS_LOG(INFO) << "Launch FusedPushMetricsKernelMod.";
    FBBuilder fbb;
    BuildPushMetricsReq(&fbb, loss, accuracy);
    uint32_t retry_time = 0;
    std::shared_ptr<std::vector<unsigned char>> push_metrics_rsp_msg = nullptr;
    do {
      if (ExitHandler::Instance().HasStopped()) {
        MS_LOG(WARNING) << "Worker has finished.";
        return true;
      }
      retry_time++;
      if (!fl::worker::HybridWorker::GetInstance().SendToServer(fbb.GetBufferPointer(), fbb.GetSize(),
                                                          fl::TcpUserCommand::kPushMetrics, &push_metrics_rsp_msg)) {
        MS_LOG(WARNING) << "Sending request for PushMetrics to server 0 failed.";
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDurationOfPushMetrics));
        continue;
      } else {
        break;
      }
    } while (retry_time < kMaxRetryTime);
    MS_LOG(INFO) << "88888888888888888888888888888888.";

    flatbuffers::Verifier verifier(push_metrics_rsp_msg->data(), push_metrics_rsp_msg->size());
    if (!verifier.VerifyBuffer<schema::ResponsePushMetrics>()) {
      MS_LOG(EXCEPTION) << "The schema of ResponsePushMetrics is invalid.";
      return false;
    }

    const schema::ResponsePushMetrics *push_metrics_rsp =
      flatbuffers::GetRoot<schema::ResponsePushMetrics>(push_metrics_rsp_msg->data());
    MS_EXCEPTION_IF_NULL(push_metrics_rsp);
    auto response_code = push_metrics_rsp->retcode();
    switch (response_code) {
      case schema::ResponseCode_SUCCEED:
      case schema::ResponseCode_OutOfTime:
        break;
      default:
        MS_LOG(WARNING) << "Launching push metrics for worker failed.";
    }

    MS_LOG(INFO) << "Push metrics for loss and accuracy success.";
    fl::worker::HybridWorker::GetInstance().SetIterationCompleted();
    return true;
  }

 private:
  template <typename T>
  void BuildPushMetricsReq(FBBuilder *fbb, T loss, T accuracy) {
    MS_EXCEPTION_IF_NULL(fbb);
    schema::RequestPushMetricsBuilder req_push_metrics_builder(*fbb);
    req_push_metrics_builder.add_loss(loss);
    req_push_metrics_builder.add_accuracy(accuracy);
    auto req_push_metrics = req_push_metrics_builder.Finish();
    fbb->Finish(req_push_metrics);
  }

  size_t total_iteration_;
};
}  // namespace kernel
}  // namespace worker
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_PUSH_METRICS_H_
