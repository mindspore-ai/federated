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

#include "common/communicator/task_executor.h"

namespace mindspore {
namespace fl {
TaskExecutor::TaskExecutor(size_t thread_num, size_t max_task_num, size_t submit_timeout)
    : submit_timeout_(submit_timeout), max_task_num_(max_task_num) {
  auto task_fun = [this]() {
    while (true) {
      std::unique_lock<std::mutex> lock(mtx_);
      if (has_stopped_) {
        return;
      }
      while (task_queue_.empty()) {
        cv_.wait_for(lock, std::chrono::milliseconds(500),
                     [this] { return has_stopped_.load() || !task_queue_.empty(); });
        if (has_stopped_) {
          return;
        }
      }
      auto task = task_queue_.front();
      task_queue_.pop();
      lock.unlock();
      task();
    }
  };
  for (size_t i = 0; i < thread_num; i++) {
    working_threads_.emplace_back(task_fun);
  }
}

void TaskExecutor::Stop() {
  if (has_stopped_) {
    return;
  }
  has_stopped_ = true;
  cv_.notify_all();
  for (auto &t : working_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
  working_threads_.clear();
  std::unique_lock<std::mutex> lock(mtx_);
  task_queue_ = std::queue<std::function<void()>>();  // clear
}

TaskExecutor::~TaskExecutor() { Stop(); }
}  // namespace fl
}  // namespace mindspore
