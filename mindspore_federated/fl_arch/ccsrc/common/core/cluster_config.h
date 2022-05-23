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

#ifndef MINDSPORE_CCSRC_PS_CORE_CLUSTER_CONFIG_H_
#define MINDSPORE_CCSRC_PS_CORE_CLUSTER_CONFIG_H_

#include <string>
#include <iostream>
#include <memory>
#include <utility>
#include <unordered_map>

#include "common/utils/log_adapter.h"
#include "common/core/node_info.h"

namespace mindspore {
namespace fl {
/*
 * Configuration information read through environment variables and configuration files, generally immutable
 */
struct ClusterConfig {
  explicit ClusterConfig() : cluster_available_timeout(900) {}
  // Timeout period for cluster preparation is 900 seconds.
  uint32_t cluster_available_timeout;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_CLUSTER_CONFIG_H_
