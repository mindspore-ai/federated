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
#include "distributed_cache/instance_context.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include "distributed_cache/distributed_cache.h"
#include "distributed_cache/redis_keys.h"
#include "common/common.h"
#include "distributed_cache/hyper_params.h"
#include "distributed_cache/common.h"
#include "distributed_cache/client_infos.h"
#include "distributed_cache/timer.h"
#include "distributed_cache/counter.h"
#include "distributed_cache/server.h"

namespace mindspore {
namespace fl {
namespace cache {
namespace {
const char *kFiledIterationNum = "iterationNum";
const char *kFiledLastIterationSuccess = "lastIterationSuccess";
const char *kFiledLastIterationResult = "lastIterationResult";
const char *kFiledRunningState = "runningState";
const char *kFiledPrimeName = "prime";

const char *kStateRunningStr = "Running";
const char *kStateDisableStr = "Disable";
const char *kStateFinishStr = "Finish";
const char *kStateStopStr = "Stop";
}  // namespace

std::string GetInstanceStateStr(const InstanceState &instance_state) {
  switch (instance_state) {
    case kStateRunning:
      return kStateRunningStr;
    case kStateDisable:
      return kStateDisableStr;
    case kStateFinish:
      return kStateFinishStr;
    case kStateStop:
      return kStateStopStr;
    default:
      return kStateRunningStr;  // default
  }
}

InstanceState GetInstanceState(const std::string &instance_state) {
  if (instance_state == kStateRunningStr) {
    return kStateRunning;
  } else if (instance_state == kStateFinishStr) {
    return kStateFinish;
  } else if (instance_state == kStateDisableStr) {
    return kStateDisable;
  }
  return kStateRunning;  // default
}

InstanceContext::InstanceContext() = default;

CacheStatus InstanceContext::InitAndSync(const std::string &fl_name, const uint64_t &recovery_iteration) {
  fl_name_ = fl_name;
  iteration_num_ = recovery_iteration;
  new_iteration_num_ = iteration_num_;
  instance_name_ = CreateNewInstanceName();
  new_instance_name_ = instance_name_;
  // sync instance name, iteration info
  auto ret = Sync();
  if (!ret.IsSuccess()) {
    return ret;
  }
  if (instance_name_ != new_instance_name_) {
    instance_name_ = new_instance_name_;
    // Change the running state to the default value when instance name changed.
    instance_state_ = InstanceState::kStateRunning;
    iteration_num_ = 1;
    new_iteration_num_ = iteration_num_;
    // Sync iteration info
    ret = Sync();
    if (!ret.IsSuccess()) {
      return ret;
    }
  }
  iteration_num_ = new_iteration_num_;

  auto fl_iteration_num = FLContext::instance()->fl_iteration_num();
  if (iteration_num_ > fl_iteration_num) {
    set_instance_state(InstanceState::kStateFinish);
  }
  return kCacheSuccess;
}

std::string InstanceContext::CreateNewInstanceName() {
  auto instance_name = "i_" + GetTimeString();
  return instance_name;
}

void InstanceContext::set_instance_state(InstanceState instance_state) {
  instance_state_ = instance_state;
  MS_LOG(INFO) << "Instance state is updated to " << cache::GetInstanceStateStr(instance_state_);
}

bool InstanceContext::HasIterationFailed(uint64_t iteration_num) const {
  if (new_iteration_num_ != iteration_num) {
    return !last_iteration_success_;
  }
  return false;
}

InstanceEventType InstanceContext::GetInstanceEventType() const {
  if (new_instance_name_ != instance_name_) {
    return kInstanceEventNewInstance;
  }
  if (new_iteration_num_ != iteration_num_) {
    return kInstanceEventNewIteration;
  }
  return kInstanceEventNone;
}

bool InstanceContext::HandleInstanceEvent() {
  if (new_instance_name_ != instance_name_) {
    OnNewInstance();
  } else if (new_iteration_num_ != iteration_num_) {
    OnNewIteration();
  }
  return true;
}

void InstanceContext::MoveToNextIterationLocal(uint64_t curr_iteration_num, bool iteration_success,
                                               const std::string &iteration_result) {
  if (instance_state_ == kStateFinish) {
    return;
  }
  uint64_t updated_iteration_num = curr_iteration_num + 1;
  // system iteration has updated
  if (updated_iteration_num == iteration_num_) {
    MS_LOG_INFO << "Update iteration num " << updated_iteration_num << " == current iteration num " << iteration_num_;
    return;
  }
  if (new_iteration_num_ == updated_iteration_num) {
    // The successful iteration result is preferred,
    // and the failed server will sync the model from the successful server when Iteration::SaveModel.
    if (!iteration_success && last_iteration_success_) {
      return;
    }
  }
  new_iteration_num_ = updated_iteration_num;
  last_iteration_success_ = iteration_success;
  last_iteration_result_ = iteration_result;
}

void InstanceContext::OnNewInstance() {
  if (new_instance_name_.empty()) {
    return;
  }
  if (instance_name_ == new_instance_name_) {
    MS_LOG_WARNING << "New instance name cannot equal to old instance name, new instance name: " << new_instance_name_;
    return;
  }
  ClearCache();
  ClearInstance();
  instance_name_ = new_instance_name_;
  new_iteration_num_ = 1;
  iteration_num_ = new_iteration_num_;
  instance_state_ = kStateRunning;
  HyperParams::Instance().SyncOnNewInstance();
  Server::Instance().Sync();
  InstanceContext::Instance().Sync();
  MS_LOG_INFO << "Handle new instance request, new instance name: " << instance_name_
              << ", iteration num reset to 1 and state reset to Running";
}

void InstanceContext::OnNewIteration() {
  if (new_iteration_num_ == iteration_num_) {
    return;
  }
  // clear cache
  ClearCache();
  // Sync server when distributed cache restart cause move to next
  (void)Server::Instance().Sync();
  auto fl_iteration_num = FLContext::instance()->fl_iteration_num();
  if (new_iteration_num_ > fl_iteration_num) {
    set_instance_state(kStateFinish);
    MS_LOG_INFO << "The instance has finished, fl_iteration_num: " << fl_iteration_num
                << ", current iteration num: " << iteration_num_;
  }
  iteration_num_ = new_iteration_num_;
}

void InstanceContext::OnStateUpdate(InstanceState new_state) {
  if (instance_state_ == new_state) {
    return;
  }
  if (new_state == kStateDisable && instance_state_ == kStateRunning) {
    set_instance_state(new_state);
    MoveToNextIterationLocal(iteration_num_, false, "Disable instance");
  } else if (new_state == kStateRunning && instance_state_ == kStateDisable) {
    set_instance_state(new_state);
  } else if (new_state == kStateStop) {
    set_instance_state(kStateStop);
  }
}

CacheStatus InstanceContext::UpdateCacheWhenCacheEmpty(const std::shared_ptr<RedisClientBase> &client) {
  if (client == nullptr) {
    return kCacheNetErr;
  }
  std::unordered_map<std::string, std::string> values = {
    {kFiledIterationNum, std::to_string(new_iteration_num_)},
    {kFiledLastIterationSuccess, std::to_string(last_iteration_success_)},
    {kFiledLastIterationResult, last_iteration_result_},
    {kFiledRunningState, std::to_string(instance_state_)},
    {kFiledPrimeName, prime_},
  };
  auto key = RedisKeys::GetInstance().InstanceStatusHash();
  auto ret = client->HMSet(key, values);
  if (!ret.IsSuccess()) {
    MS_LOG_WARNING << "Sync iteration info to distributed buffer failed";
    return ret;
  }
  auto time_window = Timer::config_expire_time_in_seconds();
  ret = client->Expire(key, time_window);
  if (!ret.IsSuccess()) {
    MS_LOG_ERROR << "Update expire time of iteration info failed";
    return ret;
  }
  return kCacheSuccess;
}

CacheStatus InstanceContext::UpdateCacheWhenNextIteration(const std::shared_ptr<RedisClientBase> &client) {
  if (client == nullptr) {
    return kCacheNetErr;
  }
  std::unordered_map<std::string, std::string> values = {
    {kFiledIterationNum, std::to_string(new_iteration_num_)},
    {kFiledLastIterationSuccess, std::to_string(last_iteration_success_)},
    {kFiledLastIterationResult, last_iteration_result_},
  };
  auto fl_iteration_num = FLContext::instance()->fl_iteration_num();
  if (new_iteration_num_ > fl_iteration_num) {
    values[kFiledRunningState] = std::to_string(InstanceState::kStateFinish);
  }
  auto key = RedisKeys::GetInstance().InstanceStatusHash();
  auto ret = client->HMSet(key, values);
  if (!ret.IsSuccess()) {
    MS_LOG_WARNING << "Sync iteration info to distributed buffer failed";
    return ret;
  }
  ret = client->Expire(key, Timer::config_expire_time_in_seconds());
  if (!ret.IsSuccess()) {
    MS_LOG_WARNING << "Update expire time of iteration info failed";
  }
  return kCacheSuccess;
}

void InstanceContext::SetPrime(const std::string &prime) {
  prime_ = prime;
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_ERROR << "Get redis client failed";
    return;
  }
  auto key = RedisKeys::GetInstance().InstanceStatusHash();
  auto ret = client->HSet(key, kFiledPrimeName, prime);
  if (!ret.IsSuccess()) {
    MS_LOG_WARNING << "Sync prime info to distributed buffer failed";
    return;
  }
}

std::string InstanceContext::GetPrime() {
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_ERROR << "Get redis client failed";
    return "";
  }
  auto key = RedisKeys::GetInstance().InstanceStatusHash();
  std::string prime;
  auto ret = client->HGet(key, kFiledPrimeName, &prime);
  if (ret.IsNil()) {
    return "";
  }
  if (!ret.IsSuccess()) {
    MS_LOG_WARNING << "Get prime info from distributed buffer failed";
    return "";
  }
  return prime;
}

CacheStatus InstanceContext::Sync(bool *is_cache_empty) {
  std::unique_lock<std::mutex> lock(lock_);
  need_sync_ = true;
  auto ret = SyncInner(is_cache_empty);
  if (ret.IsSuccess()) {
    need_sync_ = false;
  }
  return ret;
}

CacheStatus InstanceContext::SyncInstanceName(const std::shared_ptr<RedisClientBase> &client) {
  if (client == nullptr) {
    MS_LOG_ERROR << "Get redis client failed";
    return kCacheNetErr;
  }
  auto key = RedisKeys::GetInstance().InstanceNameString();
  auto ret = client->SetExNx(key, instance_name_, Timer::config_expire_time_in_seconds());
  if (ret.IsSuccess()) {
    return kCacheSuccess;
  }
  if (ret != kCacheExist) {
    return ret;
  }
  std::string instance_name_cache;
  ret = client->Get(key, &instance_name_cache);
  if (!ret.IsSuccess()) {
    MS_LOG_ERROR << "Get instance name from cache failed";
    return ret;
  }
  if (instance_name_cache.empty()) {
    return client->SetEx(key, instance_name_, Timer::config_expire_time_in_seconds());
  }
  if (instance_name_ != instance_name_cache) {
    new_instance_name_ = instance_name_cache;
  }
  (void)client->Expire(key, Timer::config_expire_time_in_seconds());
  return kCacheSuccess;
}

CacheStatus InstanceContext::SyncInstanceState(const std::shared_ptr<RedisClientBase> &client,
                                               const std::unordered_map<std::string, std::string> &values) {
  if (client == nullptr) {
    return kCacheInnerErr;
  }
  CacheStatus ret;
  auto key = RedisKeys::GetInstance().InstanceStatusHash();
  auto iteration_state = static_cast<uint64_t>(instance_state_);
  auto it = values.find(kFiledRunningState);
  if (it == values.end()) {  // default running
    iteration_state = static_cast<uint64_t>(InstanceState::kStateRunning);
  } else {
    const auto &filed_val = it->second;
    if (!Str2Uint64(it->second, &iteration_state) || iteration_state >= InstanceState::kStateMaximum) {
      MS_LOG_WARNING << "The filed value of " << key << " is invalid, filed: " << kFiledRunningState
                     << ", value: " << filed_val;
      ret = client->HSet(key, kFiledRunningState, std::to_string(instance_state_));
      if (!ret.IsSuccess()) {
        return ret;
      }
    }
  }
  if (iteration_state != static_cast<uint64_t>(instance_state_)) {
    if (instance_state_ == InstanceState::kStateFinish) {
      MS_LOG_INFO << "The instance has finished, update the finish state to the cache";
      ret = client->HSet(key, kFiledRunningState, std::to_string(instance_state_));
      if (!ret.IsSuccess()) {
        return ret;
      }
    }
    OnStateUpdate(static_cast<InstanceState>(iteration_state));
  }
  return kCacheSuccess;
}

CacheStatus InstanceContext::SyncIterationInfo(const std::shared_ptr<RedisClientBase> &client,
                                               const std::unordered_map<std::string, std::string> &values) {
  if (client == nullptr) {
    return kCacheInnerErr;
  }
  CacheStatus ret;
  auto key = RedisKeys::GetInstance().InstanceStatusHash();
  uint64_t iteration_num_cache = 0;
  auto ret_b = GetUintValue(values, kFiledIterationNum, &iteration_num_cache);
  if (!ret_b || iteration_num_cache <= 0) {
    MS_LOG_WARNING << "Get filed " << kFiledIterationNum << " from " << key << " failed";
    return UpdateCacheWhenCacheEmpty(client);
  }
  // move to next iteration, including disable, sync local to cache
  if (iteration_num_cache == iteration_num_ && new_iteration_num_ != iteration_num_cache) {
    return UpdateCacheWhenNextIteration(client);
  }
  // sync from cache to local
  if (iteration_num_cache != iteration_num_) {
    uint64_t last_iteration_success = 0;
    (void)GetUintValue(values, kFiledLastIterationSuccess, &last_iteration_success);
    std::string last_iteration_result;
    GetStrValue(values, kFiledLastIterationResult, &last_iteration_result);
    MoveToNextIterationLocal(iteration_num_cache - 1, last_iteration_success != 0, last_iteration_result);
  }
  return kCacheSuccess;
}

CacheStatus InstanceContext::SyncInner(bool *is_cache_empty) {
  if (is_cache_empty != nullptr) {
    *is_cache_empty = false;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_ERROR << "Get redis client failed";
    return kCacheNetErr;
  }
  auto ret = SyncInstanceName(client);
  if (!ret.IsSuccess()) {
    return ret;
  }
  if (new_instance_name_ != instance_name_) {
    return kCacheSuccess;
  }
  std::unordered_map<std::string, std::string> values;
  auto key = RedisKeys::GetInstance().InstanceStatusHash();
  ret = client->HGetAll(key, &values);
  if (!ret.IsSuccess()) {
    MS_LOG_ERROR << "Get iteration num from distributed buffer failed";
    return ret;
  }
  // key not exist
  if (values.empty()) {
    MS_LOG_WARNING << "Running status in distributed cache is empty";
    if (is_cache_empty != nullptr) {
      *is_cache_empty = true;
    }
    return UpdateCacheWhenCacheEmpty(client);
  }
  // sync state
  ret = SyncInstanceState(client, values);
  if (!ret.IsSuccess()) {
    return ret;
  }
  // sync iteration info
  ret = SyncIterationInfo(client, values);
  if (!ret.IsSuccess()) {
    return ret;
  }
  return kCacheSuccess;
}

void InstanceContext::NotifyNext(bool iteration_success, const std::string &iteration_result) {
  {
    std::unique_lock<std::mutex> lock(lock_);
    MoveToNextIterationLocal(iteration_num_, iteration_success, iteration_result);
  }
  (void)Sync();
}

void InstanceContext::ClearCache() {
  ClientInfos::GetInstance().ResetOnNewIteration();
  Timer::Instance().ResetOnNewIteration();
  Counter::Instance().ResetOnNewIteration();
}

void InstanceContext::ClearInstance() {
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return;
  }
  std::vector<std::string> to_rel_keys = {
    RedisKeys::GetInstance().InstanceStatusHash(),
    RedisKeys::GetInstance().HyperParamsString(),
  };
  auto instance_rel_time_in_seconds = Timer::release_expire_time_in_seconds();
  for (auto &item : to_rel_keys) {
    client->Expire(item, instance_rel_time_in_seconds);
  }
}
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
