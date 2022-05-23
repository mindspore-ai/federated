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
#ifndef MINDSPORE_FL_CACHE_ITERATION_TASK_THREAD_H
#define MINDSPORE_FL_CACHE_ITERATION_TASK_THREAD_H
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
namespace mindspore {
namespace fl {
namespace cache {
class IterationTaskThread {
 public:
  static IterationTaskThread &Instance() {
    static IterationTaskThread instance;
    return instance;
  }
  void Start();
  void Stop();
  void OnNewTask();
  // Whether waiting for new task: there is no task and no task is being handling
  bool IsTaskFinished();
  void WaitAllTaskFinish();

 private:
  std::thread task_thread_;
  std::mutex lock_;
  std::condition_variable cond_var_;
  std::atomic_bool is_stopped_ = false;
  std::atomic_bool has_task_ = false;
  std::atomic_bool handling_task_ = false;

  void TaskThreadHandle();
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_FL_CACHE_ITERATION_TASK_THREAD_H
