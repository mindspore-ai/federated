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
#include "distributed_cache/unsupervised_eval.h"
#include "armour/secure_protocol/signds.h"

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
      // cppcheck-suppress useStlAlgorithm
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
    instance_name_ = "instance_" + fl::CommUtil::GetNowTime().time_str_mill;
    FLContext::instance()->set_instance_name(instance_name_);
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
  MS_LOG(DEBUG) << "Iteration " << iteration_num << " start global timer.";
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

void Iteration::set_unsupervised_eval(float unsupervised_eval) { unsupervised_eval_ = unsupervised_eval; }

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
      end_last_iter_rsp.set_upload_accuracy(round->kernel_upload_accuracy());
      end_last_iter_rsp.set_eval_data_size(round->kernel_eval_data_size());
      auto update_model_complete_info = round->GetUpdateModelCompleteInfo();
      if (update_model_complete_info.size() != kParticipationTimeLevelNum) {
        MS_LOG(EXCEPTION) << "update_model_complete_info size is not equal 3";
        continue;
      }
      end_last_iter_rsp.set_participation_time_level1_num(update_model_complete_info[kIndexZero].second);
      end_last_iter_rsp.set_participation_time_level2_num(update_model_complete_info[kIndexOne].second);
      end_last_iter_rsp.set_participation_time_level3_num(update_model_complete_info[kIndexTwo].second);
    } else if (round->name() == "getResult") {
      end_last_iter_rsp.set_getresult_total_client_num(round->kernel_total_client_num());
      end_last_iter_rsp.set_getresult_accept_client_num(round->kernel_accept_client_num());
      end_last_iter_rsp.set_getresult_reject_client_num(round->kernel_reject_client_num());
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
  float upload_accuracy = 0.0;
  size_t eval_data_size = 0;
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

    round_client_num_map_[kGetResultTotalClientNum] += summary.getresult_total_client_num();
    round_client_num_map_[kGetResultAcceptClientNum] += summary.getresult_accept_client_num();
    round_client_num_map_[kGetResultRejectClientNum] += summary.getresult_reject_client_num();

    round_client_num_map_[kGetModelTotalClientNum] += summary.getmodel_total_client_num();
    round_client_num_map_[kGetModelAcceptClientNum] += summary.getmodel_accept_client_num();
    round_client_num_map_[kGetModelRejectClientNum] += summary.getmodel_reject_client_num();

    round_client_num_map_[kParticipationTimeLevel1] += summary.participation_time_level1_num();
    round_client_num_map_[kParticipationTimeLevel2] += summary.participation_time_level2_num();
    round_client_num_map_[kParticipationTimeLevel3] += summary.participation_time_level3_num();

    upload_loss += summary.upload_loss();
    upload_accuracy += summary.upload_accuracy();
    eval_data_size += summary.eval_data_size();
    if (summary.metrics_loss() != 0.0) {
      metrics_loss = summary.metrics_loss();
      metrics_accuracy = summary.metrics_accuracy();
    }
  }
  auto server_mode = FLContext::instance()->server_mode();
  if (server_mode == kServerModeFL || server_mode == kServerModeCloud) {
    loss_ = upload_loss;
    accuracy_ = upload_accuracy;
    size_t train_data_size = LocalMetaStore::GetInstance().value<size_t>(kCtxFedAvgTotalDataSize);
    if (train_data_size > 0) {
      loss_ = loss_ / train_data_size;
    }
    if (eval_data_size > 0) {
      accuracy_ = accuracy_ / eval_data_size;
    }
  } else {
    loss_ = metrics_loss;
    accuracy_ = metrics_accuracy;
  }
  SummarizeUnsupervisedEval();
  cache::SignDS::Instance().SummarizeSignDS();
}

void Iteration::SummarizeUnsupervisedEval() {
  std::string eval_type = FLContext::instance()->unsupervised_config().eval_type;
  if (eval_type == kNotEvalType) {
    return;
  }
  std::vector<std::string> all_eval_items;
  size_t cluster_client_num = FLContext::instance()->unsupervised_config().cluster_client_num;
  cache::Summary::Instance().GetUnsupervisedEvalItems(&all_eval_items, 0, cluster_client_num - 1);
  if (all_eval_items.empty() || all_eval_items.size() < cluster_client_num) {
    MS_LOG_INFO << "The all unsupervised eval items does not reach the unsupervised client threshold "
                << cluster_client_num << ", which is " << all_eval_items.size();
    return;
  }
  std::vector<std::vector<float>> group_ids;
  std::vector<size_t> labels;
  for (auto &item : all_eval_items) {
    UnsupervisedEvalItem unsupervised_eval_item_pb;
    auto ret = unsupervised_eval_item_pb.ParseFromString(item);
    if (!ret) {
      MS_LOG_WARNING << "Parse summary info failed";
      continue;
    }
    std::vector<float> group_id;
    for (int i = 0; i < unsupervised_eval_item_pb.eval_data_size(); i++) {
      group_id.push_back(unsupervised_eval_item_pb.eval_data(i));
    }
    auto label = cache::UnsupervisedEval::Instance().clusterArgmax(group_id);
    labels.push_back(label);
    group_ids.push_back(group_id);
  }
  float unsupervised_eval = cache::UnsupervisedEval::Instance().clusterEvaluate(group_ids, labels, eval_type);
  set_unsupervised_eval(unsupervised_eval);
  MS_LOG_INFO << "The unsupervised eval computed successfully and value is " << unsupervised_eval_ << ", eval type is "
              << eval_type;
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
  metrics.set_iteration_result(is_iteration_valid_);
  metrics.set_unsupervised_eval(unsupervised_eval_);

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
  is_iteration_valid_ = instance_context.last_iteration_valid();
  auto result = instance_context.last_iteration_result();
  // new_iteration_num is the iteration to be updated
  auto store_iteration_num = instance_context.new_iteration_num() - 1;
  if (is_iteration_valid_) {
    // This iteration of this server may have failed, or weight aggregation may not be performed.
    // Sync model from other successful servers.
    if (!Executor::GetInstance().IsIterationModelFinished(store_iteration_num)) {
      auto status = Executor::GetInstance().SyncLatestModelFromOtherServers();
      if (status.IsSuccess()) {
        return;
      }
      MS_LOG_WARNING << "Iteration " << store_iteration_num << " is invalid. Reason: " << status.StatusMessage();
      const auto &model_pair = ModelStore::GetInstance().GetLatestModel();
      ModelStore::GetInstance().StoreModelByIterNum(store_iteration_num, model_pair.second);
    } else {
      // Store the model which is successfully aggregated for this iteration.
      const auto &model = Executor::GetInstance().GetModel();
      if (model == nullptr || model->weight_data.empty() || model->weight_items.empty() ||
          !LocalMetaStore::GetInstance().verifyAggregationFeatureMap(model)) {
        MS_LOG(WARNING) << "Verify feature maps failed, iteration " << store_iteration_num
                        << " will not be stored. Use the initial iteration model instead.";
        const auto &initial_model = ModelStore::GetInstance().AssignNewModelMemory();
        ModelStore::GetInstance().StoreModelByIterNum(store_iteration_num, initial_model);
        is_iteration_valid_ = false;
        return;
      }

      ModelStore::GetInstance().StoreModelByIterNum(store_iteration_num, model);
      MS_LOG(INFO) << "Iteration " << store_iteration_num << " is successfully finished.";
    }
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
    round->InitKernelClientVisitedNum();
    round->InitKernelClientUploadLoss();
    round->InitKernelClientUploadAccuracy();
    round->InitKernelEvalDataSize();
    round->InitKernelTrainDataSize();
    round->ResetParticipationTimeAndNum();
  }
  round_client_num_map_.clear();
  set_loss(0.0f);
  set_accuracy(0.0f);
  set_unsupervised_eval(0.0f);
  std::string eval_type = FLContext::instance()->unsupervised_config().eval_type;
  if (eval_type != kNotEvalType) {
    size_t cluster_client_num = FLContext::instance()->unsupervised_config().cluster_client_num;
    cache::Summary::reset_unsupervised_eval(0, cluster_client_num - 1);
  }
  size_t &total_data_size = LocalMetaStore::GetInstance().mutable_value<size_t>(kCtxFedAvgTotalDataSize);
  total_data_size = 0;
  auto iteration_num = cache::InstanceContext::Instance().iteration_num();
  MS_LOG(DEBUG) << "Iteration " << iteration_num << " stop global timer.";
  cache::Timer::Instance().StopTimer(kGlobalTimer);
}

void Iteration::SummaryOnIterationFinish(const std::function<void()> &iteration_end_callback) {
  for (const auto &round : rounds_) {
    MS_ERROR_IF_NULL_WO_RET_VAL(round);
    round->KernelSummarize();
  }
  if (!is_iteration_valid_) {
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
  if (!data_rate_file_path_.empty() && CommUtil::CreateDirectory(data_rate_file_path_)) {
    MS_LOG(INFO) << "Create Directory :" << data_rate_file_path_ << " success.";
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
    MS_LOG(INFO) << "The data rate file path is " << data_rate_file_path_;
    auto tcp_address = server_node_->tcp_address();
    while (is_date_rate_thread_running_ && !data_rate_file_path_.empty()) {
      // record data every 60 seconds
      std::this_thread::sleep_for(std::chrono::seconds(60));
      auto time_now = std::chrono::system_clock::now();
      std::time_t tt = std::chrono::system_clock::to_time_t(time_now);
      struct tm ptm;
      (void)localtime_r(&tt, &ptm);
      std::ostringstream time_day_oss;
      time_day_oss << std::put_time(&ptm, "%Y-%m-%d");
      std::string time_day = time_day_oss.str();
      std::string data_rate_file;
      if (data_rate_file_path_ == ".") {
        data_rate_file = time_day + "_flow_server" + tcp_address + ".json";
      } else {
        data_rate_file = data_rate_file_path_ + "/" + time_day + "_flow_server" + tcp_address + ".json";
      }
      file_stream.open(data_rate_file, std::ios::out | std::ios::app);
      if (!file_stream.is_open()) {
        MS_LOG(WARNING) << data_rate_file << "is not open! Please check config file!";
        return;
      }
      std::map<uint64_t, size_t> send_datas;
      std::map<uint64_t, size_t> receive_datas;
      for (const auto &round : rounds_) {
        if (round == nullptr) {
          MS_LOG(WARNING) << "round is nullptr";
          continue;
        }
        auto send_data = round->GetSendData();
        for (const auto &it : send_data) {
          if (send_datas.find(it.first) != send_datas.end()) {
            send_datas[it.first] = send_datas[it.first] + it.second;
          } else {
            send_datas.emplace(it);
          }
        }
        auto receive_data = round->GetReceiveData();
        for (const auto &it : receive_data) {
          if (receive_datas.find(it.first) != receive_datas.end()) {
            receive_datas[it.first] = receive_datas[it.first] + it.second;
          } else {
            receive_datas.emplace(it);
          }
        }
        round->ClearData();
      }

      std::map<uint64_t, std::vector<size_t>> all_datas;
      for (auto &it : send_datas) {
        std::vector<size_t> send_and_receive_data;
        send_and_receive_data.emplace_back(it.second);
        send_and_receive_data.emplace_back(0);
        all_datas.emplace(it.first, send_and_receive_data);
      }
      for (auto &it : receive_datas) {
        if (all_datas.find(it.first) != all_datas.end()) {
          std::vector<size_t> &temp = all_datas.at(it.first);
          temp[1] = it.second;
        } else {
          std::vector<size_t> send_and_receive_data;
          send_and_receive_data.emplace_back(0);
          send_and_receive_data.emplace_back(it.second);
          all_datas.emplace(it.first, send_and_receive_data);
        }
      }
      for (auto &it : all_datas) {
        nlohmann::json js;
        auto data_time = static_cast<time_t>(it.first);
        struct tm data_tm;
        (void)localtime_r(&data_time, &data_tm);
        std::ostringstream oss_second;
        oss_second << std::put_time(&data_tm, "%Y-%m-%d %H:%M:%S");
        js["time"] = oss_second.str();
        js["send"] = it.second[0];
        js["receive"] = it.second[1];
        file_stream << js << "\n";
      }
      (void)file_stream.close();
    }
  });
  return;
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
