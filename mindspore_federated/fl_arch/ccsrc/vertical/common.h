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

#ifndef MINDSPORE_FL_ARCH_CCSRC_VERTICAL_COMMON_H_
#define MINDSPORE_FL_ARCH_CCSRC_VERTICAL_COMMON_H_

#include <map>
#include <string>
#include <numeric>
#include <climits>
#include <memory>
#include <functional>
#include <iomanip>
#include <vector>
#include "common/common.h"

namespace mindspore {
namespace fl {
struct VerticalConfig {
  // The name of the vertical server. Please refer to vertical directory files.
  std::string name;
};

enum ResponseElem { SUCCESS, FAILED };

constexpr auto KTrainer = "trainer";
constexpr auto KBobPb = "bobPb";
constexpr auto KClientPSIInit = "clientPSIInit";
constexpr auto KServerPSIInit = "serverPsiInit";
constexpr auto KAlicePbaAndBF = "alicePbaAndBF";
constexpr auto KBobAlignResult = "bobAlignResult";
constexpr auto KAliceCheck = "aliceCheck";
constexpr auto KPlainData = "plainData";
constexpr auto KDataJoin = "dataJoin";

constexpr auto KTrainerMsgType = "/trainer";
constexpr auto KBobPbMsgType = "/bobPb";
constexpr auto KClientPSIInitMsgType = "/clientPSIInit";
constexpr auto KServerPSIInitMsgType = "/serverPsiInit";
constexpr auto KAlicePbaAndBFMsgType = "/alicePbaAndBF";
constexpr auto KBobAlignResultMsgType = "/bobAlignResult";
constexpr auto KAliceCheckMsgType = "/aliceCheck";
constexpr auto KPlainDataMsgType = "/plainData";
constexpr auto KDataJoinMsgType = "/dataJoin";

constexpr size_t kRetryCommunicateTimes = 900;
constexpr size_t kSleepSecondsOfCommunicate = 1;
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_COMMON_H_
