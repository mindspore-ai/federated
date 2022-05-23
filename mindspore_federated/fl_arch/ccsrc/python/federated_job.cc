/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "common/utils/log_adapter.h"
#include "common/common.h"
#include "server/server.h"
#include "worker/fl_worker.h"
#include "scheduler/scheduler.h"
#include "python/fl_context.h"
#include "worker/kernel/start_fl_job_kernel.h"
#include "worker/kernel/update_model_kernel.h"
#include "worker/kernel/get_model_kernel.h"

namespace mindspore {
namespace fl {
using StartFLJobKernelMod = worker::kernel::StartFLJobKernelMod;
using UpdateModelKernelMod = worker::kernel::UpdateModelKernelMod;
using GetModelKernelMod = worker::kernel::GetModelKernelMod;
void FederatedJob::StartFederatedJob() {
  bool result = false;
  if (FLContext::instance()->is_server()) {
    result = StartServerAction();
  } else if (FLContext::instance()->is_scheduler()) {
    result = StartSchedulerAction();
  } else if (FLContext::instance()->is_worker()) {
    result = StartFLWorkerAction();
  }
  if (!result) {
    MS_LOG(ERROR) << "Start federated job failed.";
  }
}

bool FederatedJob::StartServerAction() {
  // Update model threshold is a certain ratio of start_fl_job threshold.
  // update_model_threshold = start_fl_job_threshold * update_model_ratio.
  size_t start_fl_job_threshold = FLContext::instance()->start_fl_job_threshold();
  float update_model_ratio = FLContext::instance()->update_model_ratio();
  size_t update_model_threshold = static_cast<size_t>(std::ceil(start_fl_job_threshold * update_model_ratio));

  std::vector<RoundConfig> rounds_config = {
    {"startFLJob", true, FLContext::instance()->start_fl_job_time_window(), true, start_fl_job_threshold},
    {"updateModel", true, FLContext::instance()->update_model_time_window(), true, update_model_threshold},
    {"getModel"},
    {"pullWeight"},
    {"pushWeight", false, 3000, true, FLContext::instance()->initial_server_num(), true},
    {"pushMetrics", false, 3000, true, 1}};

  float share_secrets_ratio = FLContext::instance()->share_secrets_ratio();
  uint64_t cipher_time_window = FLContext::instance()->cipher_time_window();
  size_t minimum_clients_for_reconstruct = FLContext::instance()->reconstruct_secrets_threshold() + 1;

  size_t exchange_keys_threshold =
    std::max(static_cast<size_t>(std::ceil(start_fl_job_threshold * share_secrets_ratio)), update_model_threshold);
  size_t get_keys_threshold =
    std::max(static_cast<size_t>(std::ceil(exchange_keys_threshold * share_secrets_ratio)), update_model_threshold);
  size_t share_secrets_threshold =
    std::max(static_cast<size_t>(std::ceil(get_keys_threshold * share_secrets_ratio)), update_model_threshold);
  size_t get_secrets_threshold =
    std::max(static_cast<size_t>(std::ceil(share_secrets_threshold * share_secrets_ratio)), update_model_threshold);
  size_t client_list_threshold = std::max(static_cast<size_t>(std::ceil(update_model_threshold * share_secrets_ratio)),
                                          minimum_clients_for_reconstruct);
  size_t push_list_sign_threshold = std::max(
    static_cast<size_t>(std::ceil(client_list_threshold * share_secrets_ratio)), minimum_clients_for_reconstruct);
  size_t get_list_sign_threshold = std::max(
    static_cast<size_t>(std::ceil(push_list_sign_threshold * share_secrets_ratio)), minimum_clients_for_reconstruct);
  std::string encrypt_type = FLContext::instance()->encrypt_type();
  if (encrypt_type == kPWEncryptType) {
    MS_LOG(INFO) << "Add secure aggregation rounds.";
    rounds_config.push_back({"exchangeKeys", true, cipher_time_window, true, exchange_keys_threshold});
    rounds_config.push_back({"getKeys", true, cipher_time_window, true, get_keys_threshold});
    rounds_config.push_back({"shareSecrets", true, cipher_time_window, true, share_secrets_threshold});
    rounds_config.push_back({"getSecrets", true, cipher_time_window, true, get_secrets_threshold});
    rounds_config.push_back({"getClientList", true, cipher_time_window, true, client_list_threshold});
    rounds_config.push_back({"reconstructSecrets", true, cipher_time_window, true, minimum_clients_for_reconstruct});
    if (FLContext::instance()->pki_verify()) {
      rounds_config.push_back({"pushListSign", true, cipher_time_window, true, push_list_sign_threshold});
      rounds_config.push_back({"getListSign", true, cipher_time_window, true, get_list_sign_threshold});
    }
  }
  if (encrypt_type == kStablePWEncryptType) {
    MS_LOG(INFO) << "Add stable secure aggregation rounds.";
    rounds_config.push_back({"exchangeKeys", true, cipher_time_window, true, exchange_keys_threshold});
    rounds_config.push_back({"getKeys", true, cipher_time_window, true, get_keys_threshold});
  }
  CipherConfig cipher_config = {share_secrets_ratio,     cipher_time_window,
                                exchange_keys_threshold, get_keys_threshold,
                                share_secrets_threshold, get_secrets_threshold,
                                client_list_threshold,   push_list_sign_threshold,
                                get_list_sign_threshold, minimum_clients_for_reconstruct};

  size_t executor_threshold = update_model_threshold;
  server::Server::GetInstance().Initialize(true, true, FLContext::instance()->fl_server_port(), rounds_config,
                                           cipher_config, executor_threshold);
  server::Server::GetInstance().Run();
  return true;
}

bool FederatedJob::StartSchedulerAction() { return Scheduler::GetInstance().Run(); }

bool FederatedJob::StartFLWorkerAction() { return worker::FLWorker::GetInstance().Run(); }

bool FederatedJob::StartFLJob(size_t data_size) { return StartFLJobKernelMod::GetInstance()->Launch(data_size); }

py::dict FederatedJob::UpdateAndGetModel(std::map<std::string, std::vector<float>> weight_datas) {
  py::dict dict_data;
  if (!UpdateModelKernelMod::GetInstance()->Launch(weight_datas)) {
    return dict_data;
  }
  return GetModelKernelMod::GetInstance()->Launch();
}
}  // namespace fl
}  // namespace mindspore
