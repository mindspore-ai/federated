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

#ifndef MINDSPORE_FEDERATED_COMMON_EXIT_HANDLER_H
#define MINDSPORE_FEDERATED_COMMON_EXIT_HANDLER_H
#include "common/utils/visible.h"

namespace mindspore {
namespace fl {
class MS_EXPORT ExitHandler {
 public:
  static ExitHandler &Instance() {
    static ExitHandler exit_handler;
    return exit_handler;
  }
  void InitSignalHandle();
  void SetStopFlag();
  bool HasStopped() const;
  int GetSignal() const;

 private:
  bool stop_flag_ = false;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FEDERATED_COMMON_EXIT_HANDLER_H
