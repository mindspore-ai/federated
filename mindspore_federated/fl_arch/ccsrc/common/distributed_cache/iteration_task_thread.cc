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
#include "distributed_cache/iteration_task_thread.h"
#include "distributed_cache/timer.h"
#include "distributed_cache/counter.h"
#include "distributed_cache/instance_context.h"
#include "common/common.h"
namespace mindspore {
namespace fl {
namespace cache {
void IterationTaskThread::Start() {
  MS_LOG_INFO << "Start thread that handles counter and timer events";
  task_thread_ = std::thread([this]() { TaskThreadHandle(); });
}
void IterationTaskThread::Stop() {
  is_stopped_ = true;
  cond_var_.notify_all();
  if (task_thread_.joinable()) {
    task_thread_.join();
  }
  MS_LOG_INFO << "End thread that handles counter and timer events";
}

void IterationTaskThread::OnNewTask() {
  if (InstanceContext::Instance().IsSafeMode()) {
    return;
  }
  std::unique_lock<std::mutex> lock(lock_);
  has_task_ = true;
  cond_var_.notify_all();
}

// Whether waiting for new task: there is no task and no task is being handling
bool IterationTaskThread::IsTaskFinished() {
  std::unique_lock<std::mutex> lock(lock_);
  return !has_task_ && !handling_task_;
}

void IterationTaskThread::WaitAllTaskFinish() {
  constexpr uint32_t kThreadSleepTime = 50;
  while (!IsTaskFinished()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kThreadSleepTime));
  }
}

void IterationTaskThread::TaskThreadHandle() {
  while (!is_stopped_) {
    std::unique_lock<std::mutex> lock(lock_);
    if (!has_task_) {
      cond_var_.wait(lock, [this]() { return is_stopped_ || has_task_; });
      if (is_stopped_) {
        return;
      }
    }
    handling_task_ = true;
    has_task_ = false;
    lock_.unlock();
    Counter::Instance().HandleEvent();
    Timer::Instance().HandleEvent();
    handling_task_ = false;
  }
}
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
