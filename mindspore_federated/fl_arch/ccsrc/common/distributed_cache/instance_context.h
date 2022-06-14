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

#ifndef MINDSPORE_CCSRC_FL_CACHE_INSTANCE_CONTEXT_H
#define MINDSPORE_CCSRC_FL_CACHE_INSTANCE_CONTEXT_H

#include <string>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "distributed_cache/cache_status.h"
#include "distributed_cache/distributed_cache.h"
#include "common/protos/comm.pb.h"
#include "common/utils/visible.h"

namespace mindspore {
namespace fl {
namespace cache {
enum InstanceState : int {
  kStateRunning = 0,  // default running
  kStateDisable = 1,
  kStateFinish = 2,
  kStateMaximum,
};

extern std::string GetInstanceStateStr(const InstanceState &instance_state);
extern InstanceState GetInstanceState(const std::string &instance_state);

enum InstanceEventType : int {
  kInstanceEventNone = 0,
  kInstanceEventNewIteration = 1,
  kInstanceEventNewInstance = 2,
};

class MS_EXPORT InstanceContext {
 public:
  static InstanceContext &Instance() {
    static InstanceContext instance;
    return instance;
  }
  InstanceContext();

  CacheStatus Sync();
  void ClearCache();
  void ClearInstance();

  CacheStatus InitAndSync(const std::string &fl_name);

  std::string fl_name() const { return fl_name_; }
  std::string instance_name() const { return instance_name_; }
  uint64_t iteration_num() const { return iteration_num_; }
  uint64_t new_iteration_num() const { return new_iteration_num_; }

  bool HasIterationFailed(uint64_t iteration_num) const;
  // used to stop AllReduce and store model
  bool last_iteration_valid() const { return last_iteration_success_; }
  std::string last_iteration_result() const { return last_iteration_result_; }

  void set_instance_state(InstanceState instance_state);
  InstanceState instance_state() const { return instance_state_; }

  // other data
  void SetPrime(const std::string &prime);
  std::string GetPrime();

  void NotifyNext(bool iteration_success, const std::string &iteration_result);

  void SetSafeMode(bool safe_mode) { in_safe_mode_ = safe_mode; }
  bool IsSafeMode() const { return in_safe_mode_; }

  InstanceEventType GetInstanceEventType() const;
  bool HandleInstanceEvent();

  static std::string CreateNewInstanceName();

 private:
  std::mutex lock_;
  bool need_sync_ = false;
  bool in_safe_mode_ = false;
  uint64_t iteration_num_ = 1;
  uint64_t new_iteration_num_ = 1;
  InstanceState instance_state_ = kStateRunning;  //
  std::string instance_name_;
  std::string new_instance_name_;
  bool last_iteration_success_ = true;
  std::string last_iteration_result_;
  std::string fl_name_;
  // other data
  std::string prime_;

  CacheStatus UpdateCacheWhenCacheEmpty(const std::shared_ptr<RedisClientBase> &client);
  CacheStatus UpdateCacheWhenNextIteration(const std::shared_ptr<RedisClientBase> &client);
  void MoveToNextIterationLocal(uint64_t curr_iteration_num, bool iteration_success,
                                const std::string &iteration_result);
  void OnStateUpdate(InstanceState new_state);
  void OnNewInstance();
  void OnNewIteration();
  CacheStatus SyncInner();
  CacheStatus SyncInstanceName(const std::shared_ptr<RedisClientBase> &client);
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_CACHE_INSTANCE_CONTEXT_H
