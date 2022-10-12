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
#include "common/exit_handler.h"
#include <csignal>

namespace mindspore {
namespace fl {
// The handler to capture the signal of SIGTERM. Normally this signal is triggered by cloud cluster manager like K8S.
namespace {
int g_signal = 0;
}
void SignalHandler(int signal) {
  if (g_signal == 0) {
    g_signal = signal;
    ExitHandler::Instance().SetStopFlag();
  }
}

void ExitHandler::InitSignalHandle() {
  (void)signal(SIGTERM, SignalHandler);
  (void)signal(SIGINT, SignalHandler);
}

void ExitHandler::SetStopFlag() { stop_flag_ = true; }

bool ExitHandler::HasStopped() const { return stop_flag_; }

int ExitHandler::GetSignal() const { return g_signal; }
}  // namespace fl
}  // namespace mindspore
