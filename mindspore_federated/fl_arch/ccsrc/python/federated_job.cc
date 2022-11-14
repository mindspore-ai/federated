/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "python/federated_job.h"
#include <vector>
#include "common/utils/log_adapter.h"
#include "common/common.h"
#include "server/server.h"
#include "worker/hybrid_worker.h"
#include "worker/cloud_worker.h"
#include "scheduler/scheduler.h"
#include "common/fl_context.h"
#include "server/model_store.h"
#include "worker/kernel/start_fl_job_kernel.h"
#include "worker/kernel/update_model_kernel.h"
#include "worker/kernel/get_model_kernel.h"
#include "worker/kernel/fused_pull_weight_kernel.h"
#include "worker/kernel/fused_push_weight_kernel.h"
#include "worker/kernel/fused_push_metrics_kernel.h"

namespace mindspore {
namespace fl {
using StartFLJobKernelMod = worker::kernel::StartFLJobKernelMod;
using UpdateModelKernelMod = worker::kernel::UpdateModelKernelMod;
using GetModelKernelMod = worker::kernel::GetModelKernelMod;
using FusedPullWeightKernelMod = worker::kernel::FusedPullWeightKernelMod;
using FusedPushWeightKernelMod = worker::kernel::FusedPushWeightKernelMod;
using FusedPushMetricsKernelMod = worker::kernel::FusedPushMetricsKernelMod;

void OnIterationEnd(const py::object &on_iteration_end_callback) {
  auto fl_name = cache::InstanceContext::Instance().fl_name();
  auto instance_name = FLContext::instance()->instance_name();
  bool iteration_valid = cache::InstanceContext::Instance().last_iteration_valid();
  auto iteration_reason = cache::InstanceContext::Instance().last_iteration_result();

  auto latest_model = server::ModelStore::GetInstance().GetLatestModel();
  auto model_iteration = latest_model.first;
  auto model = latest_model.second;
  if (model == nullptr) {
    MS_LOG_WARNING << "Failed to get latest model";
    return;
  }
  py::list feature_list;
  for (auto &weight_item : model->weight_items) {
    auto weight_py = FeatureItemPy::CreateFeatureFromModel(model, weight_item.second);
    feature_list.append(weight_py);
  }
  on_iteration_end_callback(feature_list, fl_name, instance_name, model_iteration, iteration_valid, iteration_reason);
}

void FederatedJob::StartFederatedServer(const std::vector<std::shared_ptr<FeatureItemPy>> &feature_list,
                                        const uint64_t &recovery_iteration, const py::object &after_stated_callback,
                                        const py::object &before_stopped_callback,
                                        const py::object &on_iteration_end_callback) {
  FLContext::instance()->set_ms_role(kEnvRoleOfServer);
  std::vector<InputWeight> feature_list_inner;
  feature_list_inner.reserve(feature_list.size());
  for (auto &item : feature_list) {
    // cppcheck-suppress useStlAlgorithm
    feature_list_inner.push_back(item->GetWeight());
  }
  server::FlCallback callback;
  callback.after_started = [after_stated_callback]() { after_stated_callback(); };
  callback.before_stopped = [before_stopped_callback]() { before_stopped_callback(); };
  callback.on_iteration_end = [on_iteration_end_callback]() { OnIterationEnd(on_iteration_end_callback); };
  server::Server::GetInstance().Run(feature_list_inner, recovery_iteration, callback);
}

void FederatedJob::StartFederatedScheduler() {
  FLContext::instance()->set_ms_role(kEnvRoleOfScheduler);
  Scheduler::GetInstance().Run();
}

void FederatedJob::InitFederatedWorker() {
  FLContext::instance()->set_ms_role(kEnvRoleOfWorker);
  auto server_mode = FLContext::instance()->server_mode();
  if (server_mode == kServerModeFL) {
    worker::CloudWorker::GetInstance().Init();
  } else if (server_mode == kServerModeHybrid) {
    worker::HybridWorker::GetInstance().Init();
  } else {
    MS_LOG(EXCEPTION) << server_mode << " is invalid, init federated worker failed.";
  }
}

void FederatedJob::StopFederatedWorker() {
  worker::HybridWorker::GetInstance().Stop();
  worker::CloudWorker::GetInstance().Stop();
}

bool FederatedJob::StartFLJob(size_t data_size) { return StartFLJobKernelMod::GetInstance()->Launch(data_size); }

py::dict FederatedJob::UpdateAndGetModel(std::map<std::string, std::vector<float>> weight_datas) {
  py::dict dict_data;
  if (!UpdateModelKernelMod::GetInstance()->Launch(&weight_datas)) {
    return dict_data;
  }
  return GetModelKernelMod::GetInstance()->Launch();
}

py::dict FederatedJob::PullWeight(const std::vector<std::string> &pull_weight_names) {
  return FusedPullWeightKernelMod::GetInstance()->Launch(pull_weight_names);
}

bool FederatedJob::PushWeight(const std::map<std::string, std::vector<float>> &weight_datas) {
  return FusedPushWeightKernelMod::GetInstance()->Launch(weight_datas);
}

bool FederatedJob::PushMetrics(float loss, float accuracy) {
  return FusedPushMetricsKernelMod::GetInstance()->Launch(loss, accuracy);
}
}  // namespace fl
}  // namespace mindspore
