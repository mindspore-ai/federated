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
#ifndef MINDSPORE_FL_CACHE_COUNT_H
#define MINDSPORE_FL_CACHE_COUNT_H
#include <string>
#include <unordered_map>
#include <functional>
#include <queue>
#include <memory>
#include "common/protos/comm.pb.h"
#include "distributed_cache/distributed_cache.h"

namespace mindspore {
namespace fl {
namespace cache {
class Counter {
 public:
  static Counter &Instance() {
    static Counter instance;
    return instance;
  }
  void Sync();
  //
  using CounterCallback = std::function<void()>;
  void RegisterCounter(const std::string &name, uint64_t threshold, const CounterCallback &first_callback,
                       const CounterCallback &last_callback);
  void RegisterPerServerCounter(const std::string &name, uint64_t threshold, const CounterCallback &first_callback,
                                const CounterCallback &last_callback);
  void ReinitCounter(const std::string &name, uint64_t threshold);
  //
  void ResetOnNewIteration();
  bool ReachThreshold(const std::string &name);
  bool Count(const std::string &name, bool *trigger_first, bool *trigger_last);
  void OnNotifyCountEvent(const ServerBroadcastMessage &msg);

  CacheStatus GetPerServerCountMap(const std::string &name, std::unordered_map<std::string, uint64_t> *count_map);

  bool HandleEvent();
  bool HasServerExit(const std::string &name);

 private:
  struct CounterInfo {
    uint64_t threshold = 0;
    CounterCallback first_callback = nullptr;
    CounterCallback last_callback = nullptr;
    bool server_hash_ = false;
    bool first_triggered = false;
    bool last_triggered = false;
    bool has_server_exit = false;  // when server_hash_ == True
  };
  std::unordered_map<std::string, CounterInfo> counter_map_;
  std::mutex lock_;
  std::queue<CounterCallback> task_que_;

  void HandleFirstCountEvent(CounterInfo *info, uint64_t event_iteration_num);
  void HandleLastCountEvent(CounterInfo *info, uint64_t event_iteration_num);
  bool GetCountInner(const std::shared_ptr<RedisClientBase> &client, const std::string &name, uint64_t *count);
  void SubmitEventHandle(const CounterCallback &task, uint64_t event_iteration_num);
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_FL_CACHE_COUNT_H
