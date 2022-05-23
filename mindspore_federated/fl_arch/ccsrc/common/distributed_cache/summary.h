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
#ifndef MINDSPORE_FL_CACHE_SUMMARY_H
#define MINDSPORE_FL_CACHE_SUMMARY_H
#include <string>
#include <vector>
#include "distributed_cache/cache_status.h"
namespace mindspore {
namespace fl {
namespace cache {
class Summary {
 public:
  static Summary &Instance() {
    static Summary instance;
    return instance;
  }
  static CacheStatus SubmitSummary(const std::string &summary_pb);
  static void GetAllSummaries(std::vector<std::string> *summary_pbs);

  static bool TryLockSummary(bool *has_finished);
  static void UnlockSummary();
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_CACHE_SUMMARY_H
