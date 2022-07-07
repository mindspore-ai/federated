/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "server/iteration.h"
#include <memory>
#include <vector>
#include <string>
#include <numeric>
#include <unordered_map>
#include "server/model_store.h"
#include "server/server.h"
#include "server/kernel/round/round_kernel_factory.h"
#include "distributed_cache/client_infos.h"
#include "distributed_cache/instance_context.h"
#include "distributed_cache/server.h"
#include "distributed_cache/timer.h"
#include "distributed_cache/counter.h"
#include "distributed_cache/hyper_params.h"
#include "distributed_cache/summary.h"

namespace mindspore {
namespace fl {
namespace server {
namespace {
const char *kGlobalTimer = "globalTimer";

const size_t kParticipationTimeLevelNum = 3;
const size_t kIndexZero = 0;
const size_t kIndexOne = 1;
const size_t kIndexTwo = 2;
const size_t kLastSecond = 59;
}  // namespace
Iteration::~Iteration() { Stop(); }

void Iteration::InitIteration(const std::shared_ptr<ServerNode> &server_node,
                              const std::vector<RoundConfig> &round_configs,
                              const std::vector<std::shared_ptr<CommunicatorBase>> &communicators) {
  server_node_ = server_node;
  if (communicators.empty()) {
    MS_LOG(EXCEPTION) << "Communicators for rounds is empty.";
  }
  // The time window for one iteration, which will be used in some round kernels.
  size_t iteration_time_window = 0;
  for (const RoundConfig &config : round_configs) {
    if (config.check_timeout) {
      iteration_time_window += config.time_window;
    }
  }
  MS_LOG(INFO) << "Time window for one iteration is " << iteration_time_window;
  for (const RoundConfig &config : round_configs) {
    std::shared_ptr<Round> round =
      std::make_shared<Round>(config.name, config.check_timeout, config.time_window, config.check_count,
                              config.threshold_count, config.per_server_count);
    MS_LOG(INFO) << "Add round " << config.name << ", check_timeout: " << config.check_timeout
                 << ", time window: " << config.time_window << ", check_count: " << config.check_count
                 << ", threshold: " << config.threshold_count;
    round->Initialize();
    for (auto &communicator : communicators) {
      round->RegisterMsgCallBack(communicator);
    }

    std::shared_ptr<kernel::RoundKernel> round_kernel = kernel::RoundKernelFactory::GetInstance().Create(config.name);
    if (round_kernel == nullptr) {
      MS_LOG(EXCEPTION) << "Round kernel for round " << config.name << " is not registered.";
    }
    round_kernel->InitKernelCommon(iteration_time_window);
    // For some round kernels, the threshold count should be set.
    round_kernel->InitKernel(round->threshold_count());
    round->BindRoundKernel(round_kernel);

    rounds_.push_back(round);
  }
  InitGlobalIterTimer();
  instance_name_ = FLContext::instance()->instance_name();
  if (instance_name_.empty()) {
    MS_LOG(WARNING) << "instance name is empty";
    instance_name_ = "instance_" + fl::CommUtil::GetNowTime().time_str_second;
  }
  InitConfig();
  InitEventTxtFile();
}

void Iteration::FinishIteration(bool is_last_iter_valid, const std::string &reason) {
  Executor::GetInstance().FinishIteration(is_last_iter_valid, reason);
}

void Iteration::SetIterationRunning() {
  auto iteration_num = cache::InstanceContext::Instance().iteration_num();
  MS_LOG(INFO) << "Iteration " << iteration_num << " start running.";
  start_time_ = fl::CommUtil::GetNowTime();
  MS_LOG(INFO) << "Iteration " << iteration_num << " start global timer.";
  cache::Timer::Instance().StartTimer(kGlobalTimer);
}

bool Iteration::ReInitForUpdatingHyperParams(const std::vector<RoundConfig> &updated_rounds_config) {
  for (const auto &updated_round : updated_rounds_config) {
    for (const auto &round : rounds_) {
      if (updated_round.name == round->name()) {
        MS_LOG(INFO) << "Reinitialize for round " << round->name();
        if (!round->ReInitForUpdatingHyperParams(updated_round.threshold_count, updated_round.time_window)) {
          MS_LOG(ERROR) << "Reinitializing for round " << round->name() << " failed.";
          return false;
        }
      }
    }
  }
  return true;
}

void Iteration::set_loss(float loss) { loss_ = loss; }

void Iteration::set_accuracy(float accuracy) { accuracy_ = accuracy; }

void Iteration::StartNewInstance() {
  ModelStore::GetInstance().Reset();
  if (!ReInitRounds()) {
    MS_LOG(ERROR) << "Reinitializing rounds failed.";
  }
  MS_LOG(INFO) << "Process iteration new instance successful.";
}

void Iteration::OnRoundLaunchStart() { running_round_num_++; }

void Iteration::OnRoundLaunchEnd() { running_round_num_--; }

void Iteration::WaitAllRoundsFinish() const {
  while (running_round_num_.load() != 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kThreadSleepTime));
  }
}

void Iteration::SubmitSummary() {
  IterationSummaryMsg end_last_iter_rsp;
  for (const auto &round : rounds_) {
    if (round == nullptr) {
      continue;
    }
    if (round->name() == "startFLJob") {
      end_last_iter_rsp.set_startfljob_total_client_num(round->kernel_total_client_num());
      end_last_iter_rsp.set_startfljob_accept_client_num(round->kernel_accept_client_num());
      end_last_iter_rsp.set_startfljob_reject_client_num(round->kernel_reject_client_num());
    } else if (round->name() == "updateModel") {
      end_last_iter_rsp.set_updatemodel_total_client_num(round->kernel_total_client_num());
      end_last_iter_rsp.set_updatemodel_accept_client_num(round->kernel_accept_client_num());
      end_last_iter_rsp.set_updatemodel_reject_client_num(round->kernel_reject_client_num());
      end_last_iter_rsp.set_upload_loss(round->kernel_upload_loss());
      auto update_model_complete_info = round->GetUpdateModelCompleteInfo();
      if (update_model_complete_info.size() != kParticipationTimeLevelNum) {
        MS_LOG(EXCEPTION) << "update_model_complete_info size is not equal 3";
        continue;
      }
      end_last_iter_rsp.set_participation_time_level1_num(update_model_complete_info[kIndexZero].second);
      end_last_iter_rsp.set_participation_time_level2_num(update_model_complete_info[kIndexOne].second);
      end_last_iter_rsp.set_participation_time_level3_num(update_model_complete_info[kIndexTwo].second);
    } else if (round->name() == "getModel") {
      end_last_iter_rsp.set_getmodel_total_client_num(round->kernel_total_client_num());
      end_last_iter_rsp.set_getmodel_accept_client_num(round->kernel_accept_client_num());
      end_last_iter_rsp.set_getmodel_reject_client_num(round->kernel_reject_client_num());
    }
  }
  end_last_iter_rsp.set_metrics_loss(loss_);
  end_last_iter_rsp.set_metrics_accuracy(accuracy_);
  auto status = cache::Summary::Instance().SubmitSummary(end_last_iter_rsp.SerializeAsString());
  if (!status.IsSuccess()) {
    MS_LOG_WARNING << "Failed to submit summary information to cache";
  }
}

void Iteration::GetAllSummaries() {
  std::vector<std::string> all_summary;
  cache::Summary::Instance().GetAllSummaries(&all_summary);
  MS_LOG_INFO << "Server count for summary " << all_summary.size();
  round_client_num_map_.clear();
  float upload_loss = 0.0;
  float metrics_loss = 0.0;
  float metrics_accuracy = 0.0;
  for (auto &item : all_summary) {
    IterationSummaryMsg summary;
    auto ret = summary.ParseFromString(item);
    if (!ret) {
      MS_LOG_WARNING << "Parse summary info failed";
      continue;
    }
    round_client_num_map_[kStartFLJobTotalClientNum] += summary.startfljob_total_client_num();
    round_client_num_map_[kStartFLJobAcceptClientNum] += summary.startfljob_accept_client_num();
    round_client_num_map_[kStartFLJobRejectClientNum] += summary.startfljob_reject_client_num();

    round_client_num_map_[kUpdateModelTotalClientNum] += summary.updatemodel_total_client_num();
    round_client_num_map_[kUpdateModelAcceptClientNum] += summary.updatemodel_accept_client_num();
    round_client_num_map_[kUpdateModelRejectClientNum] += summary.updatemodel_reject_client_num();

    round_client_num_map_[kGetModelTotalClientNum] += summary.getmodel_total_client_num();
    round_client_num_map_[kGetModelAcceptClientNum] += summary.getmodel_accept_client_num();
    round_client_num_map_[kGetModelRejectClientNum] += summary.getmodel_reject_client_num();

    round_client_num_map_[kParticipationTimeLevel1] += summary.participation_time_level1_num();
    round_client_num_map_[kParticipationTimeLevel2] += summary.participation_time_level2_num();
    round_client_num_map_[kParticipationTimeLevel3] += summary.participation_time_level3_num();

    upload_loss += summary.upload_loss();
    if (summary.metrics_loss() != 0.0) {
      metrics_loss = summary.metrics_loss();
      metrics_accuracy = summary.metrics_accuracy();
    }
  }
  if (FLContext::instance()->server_mode() == kServerModeFL) {
    loss_ = upload_loss;
    uint64_t update_model_threshold =
      FLContext::instance()->start_fl_job_threshold() * FLContext::instance()->update_model_ratio();
    if (update_model_threshold > 0) {
      loss_ = loss_ / update_model_threshold;
    }
  } else {
    loss_ = metrics_loss;
    accuracy_ = metrics_accuracy;
  }
}

bool Iteration::SummarizeIteration() {
  IterationMetrics metrics;
  if (!metrics.Initialize()) {
    MS_LOG(WARNING) << "Initializing metrics failed.";
    return false;
  }
  complete_time_ = fl::CommUtil::GetNowTime();
  metrics.SetInstanceName(cache::InstanceContext::Instance().instance_name());
  metrics.SetStartTime(start_time_);
  metrics.SetEndTime(complete_time_);
  metrics.set_fl_name(FLContext::instance()->fl_name());
  metrics.set_fl_iteration_num(FLContext::instance()->fl_iteration_num());
  metrics.set_cur_iteration_num(cache::InstanceContext::Instance().iteration_num());
  metrics.set_instance_state(cache::InstanceContext::Instance().instance_state());
  metrics.set_loss(loss_);
  metrics.set_accuracy(accuracy_);
  metrics.set_round_client_num_map(round_client_num_map_);
  auto iteration_valid = cache::InstanceContext::Instance().last_iteration_valid();
  metrics.set_iteration_result(iteration_valid);

  if (complete_time_.time_stamp < start_time_.time_stamp) {
    MS_LOG(ERROR) << "The complete_timestamp_: " << complete_time_.time_stamp
                  << ", start_timestamp: " << start_time_.time_stamp << ". One of them is invalid.";
    metrics.set_iteration_time_cost(UINT64_MAX);
  } else {
    metrics.set_iteration_time_cost(complete_time_.time_stamp - start_time_.time_stamp);
  }
  if (!metrics.Summarize()) {
    MS_LOG(ERROR) << "Summarizing metrics failed.";
    return false;
  }
  if (iteration_fail_num_ >= FLContext::instance()->continuous_failure_times()) {
    std::string node_role = "SERVER";
    std::string event = "Iteration failed " + std::to_string(iteration_fail_num_) + " times continuously";
    LogFailureEvent(node_role, server_node_->tcp_address(), event);
    // Finish sending one message, reset cout num to 0
    iteration_fail_num_ = 0;
  }
  return true;
}

void Iteration::InitEventTxtFile() {
  MS_LOG(DEBUG) << "Start init event txt";
  if (CommUtil::CreateDirectory(event_file_path_)) {
    MS_LOG(INFO) << "Create Directory :" << event_file_path_ << " success.";
  } else {
    MS_LOG_EXCEPTION << "Failed to create directory for event file " << event_file_path_;
  }
  std::fstream event_txt_file;
  event_txt_file.open(event_file_path_.c_str(), std::ios::out | std::ios::app);
  if (!event_txt_file.is_open()) {
    MS_LOG_EXCEPTION << "Failed to open event txt file " << event_file_path_;
  }
  event_txt_file.close();
  MS_LOG(DEBUG) << "Load event txt success!";
}

void Iteration::LogFailureEvent(const std::string &node_role, const std::string &node_address,
                                const std::string &event) {
  std::fstream event_txt_file;
  event_txt_file.open(event_file_path_, std::ios::out | std::ios::app);
  if (!event_txt_file.is_open()) {
    MS_LOG(WARNING) << "Failed to open event txt file " << event_file_path_;
    return;
  }
  std::string time = fl::CommUtil::GetNowTime().time_str_mill;
  std::string event_info =
    "nodeRole:" + node_role + "," + node_address + "," + "currentTime:" + time + "," + "event:" + event + ";";
  event_txt_file << event_info << "\n";
  (void)event_txt_file.flush();
  event_txt_file.close();
  MS_LOG(INFO) << "Process failure event success!";
}

// move next(success or fail), disable, sync with cache(new iteration number may != current iteration num + 1)
void Iteration::SaveModel() {
  auto &instance_context = cache::InstanceContext::Instance();
  auto is_iteration_valid = instance_context.last_iteration_valid();
  auto result = instance_context.last_iteration_result();
  // new_iteration_num is the iteration to be updated
  auto store_iteration_num = instance_context.new_iteration_num() - 1;
  if (is_iteration_valid) {
    // This iteration of this server may have failed, or weight aggregation may not be performed.
    // Sync model from other successful servers.
    if (!Executor::GetInstance().IsIterationModelFinished(store_iteration_num)) {
      auto status = Executor::GetInstance().SyncLatestModelFromOtherServers();
      if (!status.IsSuccess()) {
        MS_LOG_WARNING << "Iteration " << store_iteration_num << " is invalid. Reason: " << status.StatusMessage();
      }
      return;
    }
  }
  if (is_iteration_valid) {
    // Store the model which is successfully aggregated for this iteration.
    const auto &model = Executor::GetInstance().GetModel();
    if (model == nullptr || model->weight_data.empty() || model->weight_items.empty()) {
      MS_LOG(WARNING) << "Verify feature maps failed, iteration " << store_iteration_num << " will not be stored.";
      return;
    }
    ModelStore::GetInstance().StoreModelByIterNum(store_iteration_num, model);
    MS_LOG(INFO) << "Iteration " << store_iteration_num << " is successfully finished.";
  } else {
    // Store last iteration's model because this iteration is considered as invalid.
    const auto &model_pair = ModelStore::GetInstance().GetLatestModel();
    ModelStore::GetInstance().StoreModelByIterNum(store_iteration_num, model_pair.second);
    MS_LOG(WARNING) << "Iteration " << store_iteration_num << " is invalid. Reason: " << result;
  }
}

void Iteration::Reset() {
  for (auto &round : rounds_) {
    MS_ERROR_IF_NULL_WO_RET_VAL(round);
    round->Reset();
    round->InitkernelClientVisitedNum();
    round->InitkernelClientUploadLoss();
    round->ResetParticipationTimeAndNum();
  }
  round_client_num_map_.clear();
  set_loss(0.0f);
  set_accuracy(0.0f);
}

void Iteration::SummaryOnIterationFinish(const std::function<void()> &iteration_end_callback) {
  for (const auto &round : rounds_) {
    MS_ERROR_IF_NULL_WO_RET_VAL(round);
    round->KernelSummarize();
  }
  auto iteration_valid = cache::InstanceContext::Instance().last_iteration_valid();
  if (!iteration_valid) {
    iteration_fail_num_++;
  } else {
    iteration_fail_num_ = 0;
  }
  SubmitSummary();
  auto iteration_num = cache::InstanceContext::Instance().iteration_num();
  constexpr size_t retry_interval_in_ms = 200;
  size_t i = 0;
  bool has_finished = false;
  bool has_locked = false;
  constexpr size_t retry_lock_times = 5 * 5;  // retry 5s = 5*5*200ms
  // retry to acquire summary lock for 10s, until it has been acquired by other server
  for (; i < retry_lock_times; i++) {
    bool ret = cache::Summary::TryLockSummary(&has_finished, &has_locked);
    if (ret) {
      break;
    }
    if (has_finished) {
      MS_LOG_INFO << "Metrics and model checkpoint of iteration " << iteration_num
                  << " are successfully recorded by other server.";
      return;
    }
    // has been acquired by other server
    if (has_locked) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(retry_interval_in_ms));
  }
  if (i >= retry_lock_times) {
    MS_LOG_WARNING << "Failed to access the distributed cache server to acquire the summary lock";
    return;
  }
  if (has_locked) {
    bool has_expired = false;
    constexpr size_t retry_check_times = 20 * 5;  // retry 20s = 20*5*200ms
    for (; i < retry_check_times; i++) {
      cache::Summary::GetSummaryLockInfo(&has_finished, &has_expired);
      if (has_finished) {
        MS_LOG_INFO << "Metrics and model checkpoint of iteration " << iteration_num
                    << " are successfully recorded by other server.";
        return;
      }
      if (has_expired) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(retry_interval_in_ms));
    }
    if (i >= retry_check_times) {
      MS_LOG_WARNING << "Failed to access the distributed cache server to acquire info of the summary lock";
      return;
    }
    if (has_expired) {
      MS_LOG_WARNING << "Summary lock has been acquired by other server, and the job has not finished in 10s";
    }
    return;
  }
  GetAllSummaries();
  if (!SummarizeIteration()) {
    MS_LOG(WARNING) << "Summarizing iteration data failed.";
  }
  if (iteration_end_callback != nullptr) {
    iteration_end_callback();
  }
  cache::Summary::UnlockSummary();
  MS_LOG_INFO << "Metrics and model checkpoint of iteration " << iteration_num << " are successfully recorded.";
}

bool Iteration::ReInitRounds() {
  uint64_t start_fl_job_threshold = FLContext::instance()->start_fl_job_threshold();
  float update_model_ratio = FLContext::instance()->update_model_ratio();
  uint64_t update_model_threshold = static_cast<uint64_t>(std::ceil(start_fl_job_threshold * update_model_ratio));
  uint64_t start_fl_job_time_window = FLContext::instance()->start_fl_job_time_window();
  uint64_t update_model_time_window = FLContext::instance()->update_model_time_window();
  std::vector<RoundConfig> new_round_config = {
    {"startFLJob", true, start_fl_job_time_window, true, start_fl_job_threshold},
    {"updateModel", true, update_model_time_window, true, update_model_threshold}};
  if (!ReInitForUpdatingHyperParams(new_round_config)) {
    MS_LOG(ERROR) << "Reinitializing for updating hyper-parameters failed.";
    return false;
  }
  return true;
}

void Iteration::InitGlobalIterTimer() {
  auto global_time_window_in_ms = FLContext::instance()->global_iteration_time_window();
  auto time_callback = []() {
    std::string reason = "Global timer timeout! This iteration is invalid. Proceed to next iteration.";
    Iteration::GetInstance().FinishIteration(false, reason);
  };
  constexpr int msec_to_sec_times = 1000;
  int64_t global_time_window_in_seconds = static_cast<int64_t>(global_time_window_in_ms) / msec_to_sec_times;
  cache::Timer::Instance().RegisterTimer(kGlobalTimer, global_time_window_in_seconds, time_callback);
}

void Iteration::InitConfig() {
  data_rate_file_path_ = FLContext::instance()->data_rate_dir();
  if (data_rate_file_path_.empty()) {
    MS_LOG(EXCEPTION) << "Failed to get data rate file path from config file";
  }
  event_file_path_ = FLContext::instance()->failure_event_file();
  if (event_file_path_.empty()) {
    MS_LOG(EXCEPTION) << "Failed to get event file path from config file";
  }
}

void Iteration::StartThreadToRecordDataRate() {
  MS_LOG(INFO) << "Start to create a thread to record data rate";
  data_rate_thread_ = std::thread([&]() {
    std::fstream file_stream;
    MS_LOG(DEBUG) << "The data rate file path is " << data_rate_file_path_;
    auto tcp_address = server_node_->tcp_address();
    while (is_date_rate_thread_running_) {
      // record data every 60 seconds
      std::this_thread::sleep_for(std::chrono::seconds(60));
      auto now_time = fl::CommUtil::GetNowTime();
      std::string data_rate_file =
        data_rate_file_path_ + "/" + now_time.time_str_day + "_flow_server_" + tcp_address + ".json";
      file_stream.open(data_rate_file, std::ios::out | std::ios::app);
      if (!file_stream.is_open()) {
        MS_LOG(EXCEPTION) << data_rate_file << "is not open!";
        return;
      }
      std::multimap<uint64_t, size_t> send_datas;
      std::multimap<uint64_t, size_t> receive_datas;
      for (const auto &round : rounds_) {
        if (round == nullptr) {
          MS_LOG(WARNING) << "round is nullptr";
          continue;
        }
        auto send_data = round->GetSendData();
        for (const auto &it : send_data) {
          send_datas.emplace(it);
        }
        auto receive_data = round->GetReceiveData();
        for (const auto &it : receive_data) {
          receive_datas.emplace(it);
        }
        round->ClearData();
      }
      // One minute need record 60 groups of send data
      std::vector<size_t> send_data_rates(60, 0);
      for (const auto &it : send_datas) {
        uint64_t millisecond = it.first - send_datas.begin()->first;
        uint64_t second = millisecond / 1000;
        if (second > kLastSecond) {
          send_data_rates[kLastSecond] = send_data_rates[kLastSecond] + it.second;
        } else {
          send_data_rates[second] = send_data_rates[second] + it.second;
        }
      }

      // One minute need record 60 groups of receive data
      std::vector<size_t> receive_data_rates(60, 0);
      for (const auto &it : receive_datas) {
        uint64_t millisecond = it.first - receive_datas.begin()->first;
        uint64_t second = millisecond / 1000;
        if (second > kLastSecond) {
          receive_data_rates[kLastSecond] = receive_data_rates[kLastSecond] + it.second;
        } else {
          receive_data_rates[second] = receive_data_rates[second] + it.second;
        }
      }
      nlohmann::json js;
      auto minute_time = now_time.time_str_second;
      js["time"] = minute_time;
      js["send"] = send_data_rates;
      js["receive"] = receive_data_rates;
      file_stream << js << "\n";
      (void)file_stream.flush();
      file_stream.close();
    }
  });
}

void Iteration::Stop() {
  is_date_rate_thread_running_ = false;
  if (data_rate_thread_.joinable()) {
    data_rate_thread_.join();
  }
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
