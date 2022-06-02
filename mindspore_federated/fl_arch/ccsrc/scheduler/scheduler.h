/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_FEDERATED_FL_ARCH_CCSRC_SCHEDULER_SCHEDULER_H_
#define MINDSPORE_FEDERATED_FL_ARCH_CCSRC_SCHEDULER_SCHEDULER_H_

#include <memory>
#include "common/core/scheduler_node.h"
#include "python/fl_context.h"
#include "common/utils/visible.h"

namespace mindspore {
namespace fl {
class MS_EXPORT Scheduler {
 public:
  static Scheduler &GetInstance();

  bool Run();

 private:
  Scheduler() {
    if (scheduler_node_ == nullptr) {
      bool is_fl_mode = FLContext::instance()->server_mode() == kServerModeFL ||
                        FLContext::instance()->server_mode() == kServerModeHybrid;
      if (is_fl_mode) {
        scheduler_node_ = std::make_unique<fl::core::SchedulerNode>();
      }
    }
  }

  ~Scheduler() = default;
  Scheduler(const Scheduler &) = delete;
  Scheduler &operator=(const Scheduler &) = delete;
  std::unique_ptr<fl::core::SchedulerNode> scheduler_node_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_SCHEDULER_H_