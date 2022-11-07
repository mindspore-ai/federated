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

#ifndef MINDSPORE_FEDERATED_PARALLEL_FOR_H
#define MINDSPORE_FEDERATED_PARALLEL_FOR_H

#include <atomic>
#include <algorithm>
#include <memory>

#include "common/communicator/task_executor.h"

namespace mindspore {
namespace fl {

const size_t RESERVE_THREAD_NUM = 5;
const size_t CORE_THREAD_NUM = std::thread::hardware_concurrency();

struct ParallelSync {
 public:
  explicit ParallelSync(size_t thread_num_input) {
    size_t available_thread_num = CORE_THREAD_NUM - RESERVE_THREAD_NUM;
    if (available_thread_num <= 0) {
      available_thread_num = CORE_THREAD_NUM;
    }
    if (thread_num_input > 0 && thread_num_input <= CORE_THREAD_NUM) {
      thread_num_ = thread_num_input;
    } else if (thread_num_input == 0) {
      thread_num_ = available_thread_num;
    } else {
      MS_LOG(WARNING) << "Input thread num is non-available, use default: " << available_thread_num;
      thread_num_ = available_thread_num;
    }
    executor_ = std::make_shared<TaskExecutor>(thread_num_);
  }

  template <class F>
  void parallel_for(const size_t begin, const size_t end, const size_t grain_size, const F &f) {
    if (grain_size < 0) {
      MS_LOG(ERROR) << "Grain size must be large 0, but get " << grain_size;
    }
    if (begin >= end) {
      return;
    }
    if ((end - begin) < grain_size) {
      f(begin, end);
      return;
    }
    size_t chunk_size = (end - begin - 1) / thread_num_ + 1;
    chunk_size = std::max(static_cast<size_t>(grain_size), chunk_size);
    task_num_ = (end - begin - 1) / chunk_size + 1;
    std::atomic<size_t> finish_count(0);

    auto task = [&f, &begin, &end, &chunk_size, &finish_count](size_t task_id) {
      size_t local_start = begin + task_id * chunk_size;
      if (local_start < end) {
        size_t local_end = std::min(end, local_start + chunk_size);
        f(local_start, local_end);
        finish_count++;
      }
    };
    for (size_t i = 1; i < task_num_; ++i) {
      executor_->Submit(task, i);
    }
    task(0);
    while (finish_count < task_num_) {
      std::this_thread::yield();
    }
  }

  size_t get_thread_num() const { return thread_num_; }

  size_t get_task_num() const { return task_num_; }

 private:
  size_t thread_num_ = 1;
  size_t task_num_ = 1;

  std::shared_ptr<TaskExecutor> executor_;
};

}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_FEDERATED_PARALLEL_FOR_H
