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

#ifndef MINDSPORE_FL_ARCH_CCSRC_VERTICAL_MESSAGE_QUEUE_H_
#define MINDSPORE_FL_ARCH_CCSRC_VERTICAL_MESSAGE_QUEUE_H_

#include <utility>
#include <string>

#include <mutex>
#include <condition_variable>
#include <queue>
#include <deque>

#include "common/utils/log_adapter.h"

namespace mindspore {
namespace fl {
constexpr size_t kMaxQueueSize = 128;
template <typename T>
class MessageQueue {
 private:
  std::deque<T> queue_;
  std::mutex msg_mutex_;
  std::condition_variable message_received_cond_;

 public:
  void push(T data) {
    std::unique_lock<std::mutex> lock(msg_mutex_);
    if (queue_.size() >= kMaxQueueSize) {
      MS_LOG(WARNING) << "Reject the message because of over the queue size.";
      return;
    }
    queue_.push_back(data);
    message_received_cond_.notify_all();
  }

  T pop(const uint32_t &timeout) {
    std::unique_lock<std::mutex> lock(msg_mutex_);
    T ret;
    if (queue_.size() > 0) {
      ret = std::move(queue_.front());
      queue_.pop_front();
      return ret;
    }
    bool res = false;
    for (uint32_t i = 0; i < timeout; i++) {
      res = message_received_cond_.wait_for(lock, std::chrono::seconds(1), [this] { return queue_.size() > 0; });
      if (res) {
        ret = std::move(queue_.front());
        queue_.pop_front();
        return ret;
      }
    }
    if (!res) {
      MS_LOG(EXCEPTION) << "Wait for getting message timeout after " << timeout << "seconds.";
    }
    return ret;
  }
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_MESSAGE_QUEUE_H_
