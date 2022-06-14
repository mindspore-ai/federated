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

#ifndef MINDSPORE_CCSRC_FL_COMMUNICATOR_TCP_CLIENT_H_
#define MINDSPORE_CCSRC_FL_COMMUNICATOR_TCP_CLIENT_H_

#include <event2/event.h>
#include <event2/bufferevent.h>
#include <event2/thread.h>
#include <event2/bufferevent_ssl.h>

#include <functional>
#include <string>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <condition_variable>

#include "common/core/cluster_config.h"
#include "common/utils/convert_utils_base.h"
#include "common/core/comm_util.h"
#include "common/communicator/ssl_client.h"
#include "common/communicator/ssl_wrapper.h"
#include "common/constants.h"
#include "common/fl_context.h"
#include "common/communicator/tcp_message_handler.h"

namespace mindspore {
namespace fl {
class TcpClientDispatch {
 public:
  bufferevent *CreateBufferEvent();
  void Stop();
  void StartDispatch();

  static TcpClientDispatch &Instance() {
    static TcpClientDispatch instance;
    return instance;
  }

 private:
  event_base *event_base_ = nullptr;
  std::mutex event_base_mutex_;
  bool is_started_ = false;

  std::thread dispatch_thread_;
  TcpClientDispatch() = default;
  ~TcpClientDispatch() { Stop(); }
};

class TcpClient {
 public:
  using OnConnected = std::function<void()>;
  using OnDisconnected = std::function<void()>;
  using OnRead = std::function<void(const void *, size_t)>;
  using OnTimeout = std::function<void()>;
  using OnTimer = std::function<void()>;

  explicit TcpClient(const std::string &address, std::uint16_t port, NodeRole peer_role);
  virtual ~TcpClient();

  std::string GetServerAddress() const;
  void set_disconnected_callback(const OnDisconnected &disconnected);
  void set_connected_callback(const OnConnected &connected);
  bool WaitConnected(
    const uint32_t &connected_timeout = FLContext::instance()->cluster_config().cluster_available_timeout);
  void Stop();
  bool Start(uint64_t timeout_in_seconds);
  void SetMessageCallback(const TcpMessageHandler::MessageHandleFun &cb);
  bool SendMessage(const MessageMeta &meta, const Protos &protos, const void *data, size_t size);
  bool connected() const { return connected_; }

 protected:
  static void SetTcpNoDelay(const evutil_socket_t &fd);
  static void ReadCallback(struct bufferevent *bev, void *ctx);
  static void EventCallback(struct bufferevent *bev, std::int16_t events, void *ptr);
  void EventCallbackInner(struct bufferevent *bev, std::int16_t events);
  void NotifyConnected();

  std::string PeerRoleName() const;

 private:
  TcpMessageHandler message_handler_;

  OnConnected connected_callback_;
  OnDisconnected disconnected_callback_;
  OnRead read_callback_;

  std::mutex connection_mutex_;
  std::condition_variable connection_cond_;
  bufferevent *buffer_event_;

  std::string server_address_;
  std::uint16_t server_port_;
  NodeRole peer_role_;
  std::atomic<bool> connected_ = false;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMMUNICATOR_TCP_CLIENT_H_
