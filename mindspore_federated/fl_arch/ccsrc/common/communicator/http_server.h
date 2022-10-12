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

#ifndef MINDSPORE_CCSRC_FL_COMMUNICATOR_HTTP_SERVER_H_
#define MINDSPORE_CCSRC_FL_COMMUNICATOR_HTTP_SERVER_H_

#include "common/communicator/http_message_handler.h"

#include <event2/buffer.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/keyvalq_struct.h>
#include <event2/listener.h>
#include <event2/util.h>
#include <event2/thread.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <atomic>
#include <unordered_map>
#include <vector>

#include "common/communicator/http_request_handler.h"

namespace mindspore {
namespace fl {
class HttpServer {
 public:
  // Server address only support IPV4 now, and should be in format of "x.x.x.x"
  explicit HttpServer(const std::string &address, std::uint16_t port, size_t thread_num = 10)
      : server_address_(address), server_port_(port), thread_num_(thread_num), backlog_(1024), fd_(-1) {}

  ~HttpServer();

  // Return: true if success, false if failed, check log to find failure reason
  bool RegisterRoute(const std::string &url, OnRequestReceive *func);

  bool Start();
  void Stop();
  std::string address() const { return server_address_; }
  uint16_t port() const { return server_port_; }

 private:
  std::string server_address_;
  std::uint16_t server_port_;
  std::atomic<bool> has_stopped_ = false;
  size_t thread_num_;
  std::vector<std::shared_ptr<std::thread>> worker_threads_;
  std::vector<std::shared_ptr<HttpRequestHandler>> http_request_handlers;
  int32_t backlog_;
  std::unordered_map<std::string, OnRequestReceive *> request_handlers_;
  int fd_;

  bool InitServer();
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMMUNICATOR_HTTP_SERVER_H_
