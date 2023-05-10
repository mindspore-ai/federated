/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_FL_CACHE_SIGNDS_H
#define MINDSPORE_FL_CACHE_SIGNDS_H
#include <functional>
#include <string>
#include <vector>

namespace mindspore {
namespace fl {
namespace cache {

class SignDS {
 public:
  static SignDS &Instance() {
    static SignDS instance;
    return instance;
  }

  static size_t ComputRandomResponseB(const std::vector<std::string> &all_b_hat);

  static bool CheckOldVersion(const std::vector<std::string> &all_b_hat, uint64_t is_reached, size_t updatemodel_num);

  static void SummarizeSignDS();

  static void GetSignDSbHatAndReset(std::vector<std::string> *all_b_hat);

  float GetREst();

  uint64_t GetIsReached();

  static constexpr uint64_t kSignReachedThreshold = 10;
  static constexpr float kSignReductionFactor = 1.5;
  static constexpr uint64_t kSignExpansionFactor = 5;
  static constexpr float kSignInitREst = 0.00001;
  static constexpr uint64_t kSignInitReachedCount = 0;
  static constexpr uint64_t kSignInitIsNotReached = 0;
  static constexpr float kSignRREps = 5.0f;
  static constexpr auto kSignDSbHat0 = "0";
  static constexpr auto kSignDSbHat1 = "1";
  static constexpr float kSignOldRatioUpper = 0.05;
  static constexpr float kSignDSGlobalLRRatioOfNotReached = 3.0;
  static constexpr float kSignDSGlobalLRRatioOfReached = 3.0;
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_FL_CACHE_SIGNDS_H
