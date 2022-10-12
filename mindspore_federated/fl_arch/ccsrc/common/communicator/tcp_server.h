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

#ifndef MINDSPORE_CCSRC_FL_COMMUNICATOR_TCP_SERVER_H_
#define MINDSPORE_CCSRC_FL_COMMUNICATOR_TCP_SERVER_H_

#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/thread.h>
#include <event2/bufferevent_ssl.h>

#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include "common/communicator/tcp_message_handler.h"
#include "common/communicator/ssl_wrapper.h"
#include "common/core/cluster_config.h"
#include "common/utils/convert_utils_base.h"
#include "common/core/comm_util.h"
#include "common/constants.h"
#include "common/fl_context.h"

namespace mindspore {
namespace fl {
class TcpServer;
class TcpConnection {
 public:
  TcpConnection(struct bufferevent *bev, const evutil_socket_t &fd, TcpServer *server)
      : buffer_event_(bev), fd_(fd), server_(server) {}
  TcpConnection(const TcpConnection &) = delete;
  virtual ~TcpConnection();

  void OnReadHandler(const TcpMessageHandler::ReadBufferFun &read_fun);
  void InitConnection(const TcpMessageHandler::MessageHandleFun &callback);
  void SendMessage(const void *buffer, size_t num) const;
  bool SendMessage(const MessageMeta &meta, const Protos &protos, const void *data, size_t size) const;
  void SimpleResponse(const MessageMeta &meta);
  void ErrorResponse(const MessageMeta &meta, const std::string &error_msg);

  const TcpServer *GetServer() const;
  const evutil_socket_t &GetFd() const;

 protected:
  struct bufferevent *buffer_event_;
  evutil_socket_t fd_;
  TcpServer *server_;
  TcpMessageHandler tcp_message_handler_;
};

using OnServerReceiveMessage = std::function<void(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta,
                                                  const Protos &protos, const VectorPtr &data)>;

class TcpServer {
 public:
  using OnConnected = std::function<void(const TcpServer &, const TcpConnection &)>;
  using OnDisconnected = std::function<void(const TcpServer &, const TcpConnection &)>;
  using OnAccepted = std::function<std::shared_ptr<TcpConnection>(const TcpServer &)>;

  TcpServer(const std::string &address, std::uint16_t port);
  TcpServer(const TcpServer &server);
  virtual ~TcpServer();

  void SetServerCallback(const OnConnected &client_conn, const OnDisconnected &client_disconn,
                         const OnAccepted &client_accept);
  void Start();
  void Stop();
  void AddConnection(const evutil_socket_t &fd, std::shared_ptr<TcpConnection> connection);
  void RemoveConnection(const evutil_socket_t &fd);
  std::shared_ptr<TcpConnection> GetConnectionByFd(const evutil_socket_t &fd);
  OnServerReceiveMessage GetServerReceive() const;
  void SetMessageCallback(const OnServerReceiveMessage &cb);
  bool SendMessage(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const Protos &protos,
                   const void *data, size_t sizee);
  uint16_t BoundPort() const;
  std::string BoundIp() const;
  uint64_t ConnectionNum() const;
  const std::map<evutil_socket_t, std::shared_ptr<TcpConnection>> &Connections() const;

 protected:
  void Init();
  void StartDispatch();

  static void ListenerCallback(struct evconnlistener *listener, evutil_socket_t socket, struct sockaddr *saddr,
                               int socklen, void *server);
  void ListenerCallbackInner(evutil_socket_t socket, struct sockaddr *saddr);
  static void SignalCallback(evutil_socket_t sig, std::int16_t events, void *server);
  static void SignalCallbackInner(void *server);
  static void ReadCallback(struct bufferevent *, void *connection);
  static void ReadCallbackInner(struct bufferevent *, void *connection);
  static void EventCallback(struct bufferevent *, std::int16_t events, void *server);
  static void EventCallbackInner(struct bufferevent *, std::int16_t events, void *server);
  static void SetTcpNoDelay(const evutil_socket_t &fd);
  std::shared_ptr<TcpConnection> onCreateConnection(struct bufferevent *bev, const evutil_socket_t &fd);

  struct event_base *event_base_;
  struct evconnlistener *listener_;
  std::string server_address_;
  std::uint16_t server_port_;
  std::atomic<bool> is_stop_;

  std::map<evutil_socket_t, std::shared_ptr<TcpConnection>> connections_;
  OnConnected client_connection_;
  OnDisconnected client_disconnection_;
  OnAccepted client_accept_;
  std::mutex connection_mutex_;
  OnServerReceiveMessage message_callback_;
  uint64_t max_connection_;

  bool is_started_ = false;
  std::thread dispatch_thread_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMMUNICATOR_TCP_SERVER_H_
