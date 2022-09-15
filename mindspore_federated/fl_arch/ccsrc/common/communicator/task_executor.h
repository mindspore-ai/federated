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

#ifndef MINDSPORE_CCSRC_FL_COMMUNICATOR_TASK_EXECUTOR_H_
#define MINDSPORE_CCSRC_FL_COMMUNICATOR_TASK_EXECUTOR_H_

#include <functional>
#include <queue>
#include <mutex>
#include <vector>
#include <thread>
#include <condition_variable>
#include <atomic>

#include "common/utils/log_adapter.h"
#include "common/constants.h"

namespace mindspore {
namespace fl {
/* This class can submit tasks in multiple threads
 * example:
 * void TestTaskExecutor() {
 *   std::cout << "Execute in one thread";
 * }
 *
 * TaskExecutor executor(10); // 10 threads
 * executor.Submit(TestTaskExecutor, this); // Submit task
 */
class TaskExecutor {
 public:
  explicit TaskExecutor(size_t thread_num, size_t max_task_num = kMaxTaskNum,
                        size_t submit_timeout = kSubmitTimeOutInMs);
  ~TaskExecutor();

  // If the number of submitted tasks is greater than the size of the queue, it will block the submission of subsequent
  // tasks until timeout.
  template <typename Fun, typename... Args>
  bool Submit(Fun &&function, Args &&...args) {
    constexpr int64_t kSubmitTaskIntervalInMs = 1;
    auto callee = std::bind(function, args...);
    std::function<void()> task = [callee]() -> void { callee(); };
    for (size_t i = 0; i < submit_timeout_; i++) {
      std::unique_lock<std::mutex> lock(mtx_);
      if (has_stopped_) {
        MS_LOG(INFO) << "Submit task failed, task executor has stopped";
        return false;
      }
      if (task_queue_.size() < max_task_num_) {
        task_queue_.push(task);
        lock.unlock();
        cv_.notify_all();
        return true;
      }
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(kSubmitTaskIntervalInMs));
    }
    MS_LOG(WARNING) << "Submit task failed after " << submit_timeout_ << " ms.";
    return false;
  }

  void Stop();

 private:
  std::atomic_bool has_stopped_ = false;

  // The timeout period of the task submission, in milliseconds. default timeout is 3000 milliseconds.
  size_t submit_timeout_;

  // The maximum number of tasks that can be submitted to the task queue, If the number of submitted tasks exceeds this
  // max_task_num_, the Submit function will block.Until the current number of tasks is less than max task num,or
  // timeout.
  size_t max_task_num_;

  std::mutex mtx_;
  std::condition_variable cv_;

  std::vector<std::thread> working_threads_;
  std::queue<std::function<void()>> task_queue_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMMUNICATOR_TASK_EXECUTOR_H_
