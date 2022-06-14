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

#include "server/round.h"
#include <memory>
#include <string>
#include "server/server.h"
#include "server/iteration.h"
#include "distributed_cache/counter.h"
#include "distributed_cache/timer.h"
#include "distributed_cache/common.h"
#include "distributed_cache/instance_context.h"
#include "server/kernel/round/update_model_kernel.h"

namespace mindspore {
namespace fl {
namespace server {
class Server;
class Iteration;
std::atomic<uint32_t> kPrintTimes = 0;
const uint32_t kPrintTimesThreshold = 3000;
Round::Round(const std::string &name, bool check_timeout, uint64_t time_window, bool check_count,
             uint64_t threshold_count, bool per_server_count)
    : name_(name),
      check_timeout_(check_timeout),
      time_window_(time_window),
      check_count_(check_count),
      threshold_count_(threshold_count),
      per_server_count_(per_server_count) {}

void Round::RegisterMsgCallBack(const std::shared_ptr<CommunicatorBase> &communicator) {
  MS_EXCEPTION_IF_NULL(communicator);
  MS_LOG(INFO) << "Round " << name_ << " register message callback.";
  communicator->RegisterRoundMsgCallback(
    name_, [this](const std::shared_ptr<MessageHandler> &message) { LaunchRoundKernel(message); });
}

void Round::Initialize() {
  MS_LOG(INFO) << "Round " << name_ << " start initialize.";
  if (check_timeout_) {
    auto time_callback = [this]() {
      std::string reason = "Round " + name_ + " timeout! This iteration is invalid. Proceed to next iteration.";
      Iteration::GetInstance().FinishIteration(false, reason);
    };
    constexpr int msec_to_sec_times = 1000;
    int64_t time_window_in_seconds = static_cast<int64_t>(time_window_) / msec_to_sec_times;
    cache::Timer::Instance().RegisterTimer(name_, time_window_in_seconds, time_callback);
  }
  // Set counter event callbacks for this round if the round kernel is stateful.
  if (check_count_) {
    auto first_count_handler = [this]() { OnFirstCountEvent(); };
    auto last_count_handler = [this]() { OnLastCountEvent(); };
    if (per_server_count_) {
      cache::Counter::Instance().RegisterPerServerCounter(name_, static_cast<int64_t>(threshold_count_),
                                                          first_count_handler, last_count_handler);
    } else {
      cache::Counter::Instance().RegisterCounter(name_, static_cast<int64_t>(threshold_count_), first_count_handler,
                                                 last_count_handler);
    }
  }
}

bool Round::ReInitForUpdatingHyperParams(uint64_t updated_threshold_count, uint64_t updated_time_window) {
  time_window_ = updated_time_window;
  threshold_count_ = updated_threshold_count;
  if (check_count_) {
    cache::Counter::Instance().ReinitCounter(name_, static_cast<int64_t>(threshold_count_));
  }
  if (check_timeout_) {
    cache::Timer::Instance().ReinitTimer(name_, static_cast<int64_t>(time_window_));
  }
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_, false);
  kernel_->InitKernel(threshold_count_);
  return true;
}

void Round::BindRoundKernel(const std::shared_ptr<kernel::RoundKernel> &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  kernel_ = kernel;
}

void Round::LaunchRoundKernel(const std::shared_ptr<MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  MS_ERROR_IF_NULL_WO_RET_VAL(kernel_);

  std::string reason = "";
  if (!IsServerAvailable(&reason)) {
    if (!message->SendResponse(reason.c_str(), reason.size())) {
      MS_LOG(WARNING) << "Sending response failed.";
      return;
    }
    return;
  }
  Iteration::GetInstance().OnRoundLaunchStart();
  try {
    bool ret = kernel_->Launch(reinterpret_cast<const uint8_t *>(message->data()), message->len(), message);
    if (!ret) {
      MS_LOG(DEBUG) << "Launching round kernel of round " + name_ + " failed.";
    }
  } catch (const cache::DistributedCacheUnavailable &) {
    if (kPrintTimes % kPrintTimesThreshold == 0) {
      MS_LOG(WARNING) << "The distributed cache is not available, please retry " << name_ << " later.";
      kPrintTimes = 0;
    }
    kPrintTimes += 1;
    if (!message->HasSentResponse()) {
      reason = kJobNotAvailable;
      if (!message->SendResponse(reason.c_str(), reason.size())) {
        MS_LOG(WARNING) << "Sending response failed.";
      }
    }
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Catch exception when handle job " << name_ << ": " << e.what();
    reason = kJobNotAvailable;
    if (!message->HasSentResponse()) {
      reason = kServerInnerError;
      if (!message->SendResponse(reason.c_str(), reason.size())) {
        MS_LOG(WARNING) << "Sending response failed.";
      }
    }
  }
  Iteration::GetInstance().OnRoundLaunchEnd();

  auto time = fl::CommUtil::GetNowTime().time_stamp;
  kernel_->RecordReceiveData(std::make_pair(time, message->len()));
}

void Round::Reset() {
  MS_ERROR_IF_NULL_WO_RET_VAL(kernel_);
  (void)kernel_->Reset();
}

const std::string &Round::name() const { return name_; }

size_t Round::threshold_count() const { return threshold_count_; }

bool Round::check_timeout() const { return check_timeout_; }

size_t Round::time_window() const { return time_window_; }

void Round::OnFirstCountEvent() {
  MS_ERROR_IF_NULL_WO_RET_VAL(kernel_);
  MS_LOG(INFO) << "Round " << name_ << " first count event is triggered.";
  // The timer starts only after the first count event is triggered by DistributedCountService.
  if (check_timeout_) {
    cache::Timer::Instance().StartTimer(name_);
  }
  // Some kernels override the OnFirstCountEvent method.
  kernel_->OnFirstCountEvent();
}

void Round::OnLastCountEvent() {
  MS_ERROR_IF_NULL_WO_RET_VAL(kernel_);
  MS_LOG(INFO) << "Round " << name_ << " last count event is triggered.";
  // Same as the first count event, the timer must be stopped by DistributedCountService.
  if (check_timeout_) {
    cache::Timer::Instance().StopTimer(name_);
  }
  // Some kernels override the OnLastCountEvent method.
  kernel_->OnLastCountEvent();
}

bool Round::IsServerAvailable(std::string *reason) {
  MS_ERROR_IF_NULL_W_RET_VAL(reason, false);
  auto &context = cache::InstanceContext::Instance();
  auto state = context.instance_state();
  // After one instance is completed, the model should be accessed by clients.
  if (state == cache::InstanceState::kStateFinish && name_ == "getModel") {
    return true;
  }

  // If the server state is Disable or Finish, refuse the request.
  if (state == cache::InstanceState::kStateDisable || state == cache::InstanceState::kStateFinish) {
    if (kPrintTimes % kPrintTimesThreshold == 0) {
      MS_LOG(WARNING) << "The server's training job is disabled or finished, please retry " + name_ + " later.";
      kPrintTimes = 0;
    }
    kPrintTimes += 1;
    *reason = kJobNotAvailable;
    return false;
  }

  // If the server is still in safemode, reject the request.
  if (context.IsSafeMode()) {
    if (kPrintTimes % kPrintTimesThreshold == 0) {
      MS_LOG(WARNING) << "The cluster is still in safemode, please retry " << name_ << " later.";
      kPrintTimes = 0;
    }
    kPrintTimes += 1;
    *reason = kClusterSafeMode;
    return false;
  }
  if (!cache::DistributedCacheLoader::Instance().available() && name_ != "getModel") {
    if (kPrintTimes % kPrintTimesThreshold == 0) {
      MS_LOG(WARNING) << "The distributed cache is not available, please retry " << name_ << " later.";
      kPrintTimes = 0;
    }
    kPrintTimes += 1;
    *reason = kJobNotAvailable;
    return false;
  }
  return true;
}

void Round::KernelSummarize() {
  MS_ERROR_IF_NULL_WO_RET_VAL(kernel_);
  (void)kernel_->Summarize();
}

size_t Round::kernel_total_client_num() const { return kernel_->total_client_num(); }

size_t Round::kernel_accept_client_num() const { return kernel_->accept_client_num(); }

size_t Round::kernel_reject_client_num() const { return kernel_->reject_client_num(); }

void Round::InitkernelClientVisitedNum() { kernel_->InitClientVisitedNum(); }

void Round::InitkernelClientUploadLoss() { kernel_->InitClientUploadLoss(); }

float Round::kernel_upload_loss() const { return kernel_->upload_loss(); }

std::vector<std::pair<uint64_t, uint32_t>> Round::GetUpdateModelCompleteInfo() const {
  if (name_ == kUpdateModelKernel) {
    auto update_model_model_ptr = std::dynamic_pointer_cast<kernel::UpdateModelKernel>(kernel_);
    MS_EXCEPTION_IF_NULL(update_model_model_ptr);
    return update_model_model_ptr->GetCompletePeriodRecord();
  } else {
    MS_LOG(EXCEPTION) << "The kernel is not updateModel";
    return {};
  }
}

void Round::ResetParticipationTimeAndNum() {
  if (name_ == kUpdateModelKernel) {
    auto update_model_kernel_ptr = std::dynamic_pointer_cast<kernel::UpdateModelKernel>(kernel_);
    MS_ERROR_IF_NULL_WO_RET_VAL(update_model_kernel_ptr);
    update_model_kernel_ptr->ResetParticipationTimeAndNum();
  }
  return;
}

std::multimap<uint64_t, size_t> Round::GetSendData() const { return kernel_->GetSendData(); }

std::multimap<uint64_t, size_t> Round::GetReceiveData() const { return kernel_->GetReceiveData(); }

void Round::ClearData() { return kernel_->ClearData(); }
}  // namespace server
}  // namespace fl
}  // namespace mindspore
