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

#include "common/communicator/tcp_server.h"

#include <arpa/inet.h>
#include <event2/buffer.h>
#include <event2/buffer_compat.h>
#include <event2/bufferevent.h>
#include <event2/event.h>
#include <event2/listener.h>
#include <event2/util.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <csignal>
#include <utility>

namespace mindspore {
namespace fl {
TcpConnection::~TcpConnection() { bufferevent_free(buffer_event_); }

void TcpConnection::InitConnection(const TcpMessageHandler::MessageHandleFun &callback) {
  tcp_message_handler_.SetCallback(callback);
}

void TcpConnection::OnReadHandler(const TcpMessageHandler::ReadBufferFun &read_fun) {
  tcp_message_handler_.ReceiveMessage(read_fun);
}

void TcpConnection::SendMessage(const void *buffer, size_t num) const {
  MS_EXCEPTION_IF_NULL(buffer);
  MS_EXCEPTION_IF_NULL(buffer_event_);
  bufferevent_lock(buffer_event_);
  if (bufferevent_write(buffer_event_, buffer, num) == -1) {
    MS_LOG(ERROR) << "Write message to buffer event failed!";
  }
  bufferevent_unlock(buffer_event_);
}

const TcpServer *TcpConnection::GetServer() const { return server_; }

const evutil_socket_t &TcpConnection::GetFd() const { return fd_; }

bool TcpConnection::SendMessage(const MessageMeta &meta, const Protos &protos, const void *data, size_t size) const {
  MS_EXCEPTION_IF_NULL(buffer_event_);
  MS_EXCEPTION_IF_NULL(data);
  bufferevent_lock(buffer_event_);
  bool res = true;
  std::string meta_data = meta.SerializeAsString();
  MessageHeader header;
  header.message_proto_ = protos;
  header.message_meta_length_ = SizeToUint(meta_data.size());
  header.message_length_ = size + header.message_meta_length_;

  if (bufferevent_write(buffer_event_, &header, sizeof(header)) == -1) {
    MS_LOG(ERROR) << "Event buffer add header failed!";
    res = false;
  }
  if (bufferevent_write(buffer_event_, meta_data.data(), meta_data.size()) == -1) {
    MS_LOG(ERROR) << "Event buffer add protobuf data failed!";
    res = false;
  }
  if (bufferevent_write(buffer_event_, data, size) == -1) {
    MS_LOG(ERROR) << "Event buffer add protobuf data failed!";
    res = false;
  }
  int result = bufferevent_flush(buffer_event_, EV_READ | EV_WRITE, BEV_FLUSH);
  if (result < 0) {
    bufferevent_unlock(buffer_event_);
    MS_LOG(EXCEPTION) << "Bufferevent flush failed!";
  }
  bufferevent_unlock(buffer_event_);
  return res;
}

void TcpConnection::SimpleResponse(const MessageMeta &meta) {
  uint64_t void_data = 0;
  if (!SendMessage(meta, Protos::RAW, &void_data, sizeof(void_data))) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
}

void TcpConnection::ErrorResponse(const MessageMeta &meta, const std::string &error_msg) {
  MessageMeta new_meta = meta;
  if (!error_msg.empty()) {
    new_meta.set_response_error(error_msg);
  }
  uint64_t void_data = 0;
  if (!SendMessage(new_meta, Protos::RAW, &void_data, sizeof(void_data))) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
}

TcpServer::TcpServer(const std::string &address, std::uint16_t port)
    : event_base_(nullptr),
      listener_(nullptr),
      server_address_(std::move(address)),
      server_port_(port),
      is_stop_(true),
      max_connection_(0) {}

TcpServer::~TcpServer() { Stop(); }

void TcpServer::SetServerCallback(const OnConnected &client_conn, const OnDisconnected &client_disconn,
                                  const OnAccepted &client_accept) {
  this->client_connection_ = client_conn;
  this->client_disconnection_ = client_disconn;
  this->client_accept_ = client_accept;
}

void TcpServer::Init() {
  if (FLContext::instance()->enable_ssl()) {
    MS_LOG(INFO) << "Load ssl.";
    SSLWrapper::GetInstance().InitSSL();
  }
  int result = evthread_use_pthreads();
  if (result != 0) {
    MS_LOG(EXCEPTION) << "Use event pthread failed!";
  }

  is_stop_ = false;
  event_base_ = event_base_new();
  MS_EXCEPTION_IF_NULL(event_base_);
  if (!CommUtil::CheckIp(server_address_)) {
    MS_LOG(EXCEPTION) << "The tcp server ip:" << server_address_ << " is illegal!";
  }
  max_connection_ = FLContext::instance()->max_connection_num();
  MS_LOG(INFO) << "The max connection is:" << max_connection_;

  struct sockaddr_in sin {};
  if (memset_s(&sin, sizeof(sin), 0, sizeof(sin)) != EOK) {
    MS_LOG(EXCEPTION) << "Initialize sockaddr_in failed!";
  }
  sin.sin_family = AF_INET;
  sin.sin_port = htons(server_port_);
  sin.sin_addr.s_addr = inet_addr(server_address_.c_str());

  listener_ = evconnlistener_new_bind(event_base_, ListenerCallback, reinterpret_cast<void *>(this),
                                      LEV_OPT_REUSEABLE | LEV_OPT_CLOSE_ON_FREE, -1,
                                      reinterpret_cast<struct sockaddr *>(&sin), sizeof(sin));
  if (listener_ == nullptr) {
    MS_LOG(EXCEPTION) << "bind ip & port failed. please check.";
  }

  if (server_port_ == 0) {
    struct sockaddr_in sin_bound {};
    if (memset_s(&sin, sizeof(sin_bound), 0, sizeof(sin_bound)) != EOK) {
      MS_LOG(EXCEPTION) << "Initialize sockaddr_in failed!";
    }
    socklen_t addr_len = sizeof(struct sockaddr_in);
    if (getsockname(evconnlistener_get_fd(listener_), (struct sockaddr *)&sin_bound, &addr_len) != 0) {
      MS_LOG(EXCEPTION) << "Get sock name failed!";
    }
    server_port_ = htons(sin_bound.sin_port);
  }
}

void TcpServer::StartDispatch() {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  if (is_started_) {
    return;
  }
  if (dispatch_thread_.joinable()) {
    dispatch_thread_.join();
  }
  auto dispatch_fun = [this]() {
    is_started_ = true;
    auto ret = event_base_dispatch(event_base_);
    is_started_ = false;
    if (ret == 0) {
      MS_LOG_INFO << "Event base dispatch and exit success!";
    } else if (ret == 1) {
      MS_LOG_INFO << "Event base dispatch failed with no events pending or active!";
    } else if (ret == -1) {
      MS_LOG_WARNING << "Event base dispatch failed with error occurred!";
    } else if (ret < -1) {
      MS_LOG_WARNING << "Event base dispatch with unexpected error code!";
    }
  };
  dispatch_thread_ = std::thread(dispatch_fun);
}

void TcpServer::Start() {
  Init();
  StartDispatch();
}

void TcpServer::Stop() {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  MS_LOG(INFO) << "Stop tcp server!";
  if (is_started_ && event_base_ != nullptr) {
    int ret = event_base_loopbreak(event_base_);
    if (ret != 0) {
      MS_LOG(ERROR) << "Event base loop break failed!";
    }
  }
  connections_.clear();
  if (dispatch_thread_.joinable()) {
    dispatch_thread_.join();
  }
  if (listener_ != nullptr) {
    evconnlistener_free(listener_);
    listener_ = nullptr;
  }
  if (event_base_ != nullptr) {
    event_base_free(event_base_);
    event_base_ = nullptr;
  }
}

void TcpServer::AddConnection(const evutil_socket_t &fd, std::shared_ptr<TcpConnection> connection) {
  MS_EXCEPTION_IF_NULL(connection);
  std::lock_guard<std::mutex> lock(connection_mutex_);
  connections_.insert(std::make_pair(fd, connection));
}

void TcpServer::RemoveConnection(const evutil_socket_t &fd) {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  MS_LOG(INFO) << "Remove connection fd: " << fd;
  connections_.erase(fd);
}

std::shared_ptr<TcpConnection> TcpServer::GetConnectionByFd(const evutil_socket_t &fd) { return connections_[fd]; }

void TcpServer::ListenerCallback(struct evconnlistener *, evutil_socket_t fd, struct sockaddr *sockaddr, int,
                                 void *const data) {
  try {
    auto server = reinterpret_cast<class TcpServer *>(data);
    MS_EXCEPTION_IF_NULL(server);
    server->ListenerCallbackInner(fd, sockaddr);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpServer::ListenerCallbackInner(evutil_socket_t fd, struct sockaddr *sockaddr) {
  auto base = reinterpret_cast<struct event_base *>(event_base_);
  MS_EXCEPTION_IF_NULL(base);
  MS_EXCEPTION_IF_NULL(sockaddr);

  if (ConnectionNum() >= max_connection_) {
    MS_LOG(WARNING) << "The current connection num:" << ConnectionNum() << " is greater or equal to "
                    << max_connection_;
    return;
  }

  struct bufferevent *bev = nullptr;

  if (!FLContext::instance()->enable_ssl()) {
    MS_LOG(INFO) << "SSL is disable.";
    bev = bufferevent_socket_new(base, fd, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  } else {
    MS_LOG(INFO) << "Enable ssl support.";
    SSL *ssl = SSL_new(SSLWrapper::GetInstance().GetSSLCtx());
    MS_EXCEPTION_IF_NULL(ssl);
    bev = bufferevent_openssl_socket_new(base, fd, ssl, BUFFEREVENT_SSL_ACCEPTING,
                                         BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  }
  if (bev == nullptr) {
    MS_LOG(ERROR) << "Error constructing buffer event!";
    int ret = event_base_loopbreak(base);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "event base loop break failed!";
    }
    return;
  }

  std::shared_ptr<TcpConnection> conn = onCreateConnection(bev, fd);
  MS_EXCEPTION_IF_NULL(conn);
  SetTcpNoDelay(fd);
  AddConnection(fd, conn);
  // TcpConnection shared_ptr cannot hold itself
  std::weak_ptr<TcpConnection> conn_weak = conn;
  conn->InitConnection([this, conn_weak](const MessageMeta &meta, const Protos &protos, const VectorPtr &data) {
    auto conn = conn_weak.lock();
    if (!conn) {
      return;
    }
    OnServerReceiveMessage on_server_receive = GetServerReceive();
    if (on_server_receive) {
      on_server_receive(conn, meta, protos, data);
    }
  });
  bufferevent_setcb(bev, TcpServer::ReadCallback, nullptr, TcpServer::EventCallback,
                    reinterpret_cast<void *>(conn.get()));
  MS_LOG(INFO) << "A client is connected, fd is " << fd;
  if (bufferevent_enable(bev, EV_READ | EV_WRITE) == -1) {
    MS_LOG(EXCEPTION) << "Buffer event enable read and write failed!";
  }
}

std::shared_ptr<TcpConnection> TcpServer::onCreateConnection(struct bufferevent *bev, const evutil_socket_t &fd) {
  MS_EXCEPTION_IF_NULL(bev);
  std::shared_ptr<TcpConnection> conn = nullptr;
  if (client_accept_) {
    conn = (client_accept_(*this));
  } else {
    conn = std::make_shared<TcpConnection>(bev, fd, this);
  }

  return conn;
}

OnServerReceiveMessage TcpServer::GetServerReceive() const { return message_callback_; }

void TcpServer::SignalCallback(evutil_socket_t, std::int16_t, void *const data) {
  try {
    SignalCallbackInner(data);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpServer::SignalCallbackInner(void *const data) {
  MS_EXCEPTION_IF_NULL(data);
  auto server = reinterpret_cast<class TcpServer *>(data);
  struct event_base *base = server->event_base_;
  MS_EXCEPTION_IF_NULL(base);
  struct timeval delay = {0, 0};
  MS_LOG(ERROR) << "Caught an interrupt signal; exiting cleanly in 0 seconds.";
  if (event_base_loopexit(base, &delay) == -1) {
    MS_LOG(ERROR) << "Event base loop exit failed.";
  }
}

void TcpServer::ReadCallback(struct bufferevent *bev, void *const connection) {
  try {
    ReadCallbackInner(bev, connection);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpServer::ReadCallbackInner(struct bufferevent *bev, void *const connection) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(connection);

  auto conn = static_cast<class TcpConnection *>(connection);
  struct evbuffer *buf = bufferevent_get_input(bev);
  MS_EXCEPTION_IF_NULL(buf);
  auto read_fun = [buf](void *data, size_t max_size) -> size_t {
    auto ret = evbuffer_remove(buf, data, max_size);
    if (ret < 0) {
      return 0;
    }
    return static_cast<size_t>(ret);
  };
  conn->OnReadHandler(read_fun);
}

void TcpServer::EventCallback(struct bufferevent *bev, std::int16_t events, void *const data) {
  try {
    EventCallbackInner(bev, events, data);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpServer::EventCallbackInner(struct bufferevent *bev, std::int16_t events, void *const data) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(data);
  struct evbuffer *output = bufferevent_get_output(bev);
  MS_EXCEPTION_IF_NULL(output);
  auto conn = static_cast<class TcpConnection *>(data);
  auto srv = const_cast<TcpServer *>(conn->GetServer());
  MS_EXCEPTION_IF_NULL(srv);

  if (events & BEV_EVENT_EOF) {
    MS_LOG(INFO) << "BEV_EVENT_EOF event is trigger!";
    // Notify about disconnection
    if (srv->client_disconnection_) {
      srv->client_disconnection_(*srv, *conn);
    }
    // Free connection structures
    srv->RemoveConnection(conn->GetFd());
  } else if (events & BEV_EVENT_ERROR) {
    MS_LOG(WARNING) << "BEV_EVENT_ERROR event is trigger!";
    if (FLContext::instance()->enable_ssl()) {
      uint64_t err = bufferevent_get_openssl_error(bev);
      MS_LOG(DEBUG) << "The error number is:" << err;

      MS_LOG(DEBUG) << "Error message:" << ERR_reason_error_string(err)
                    << ", the error lib:" << ERR_lib_error_string(err)
                    << ", the error func:" << ERR_func_error_string(err);

      MS_LOG(WARNING) << "Tcp server filed!";
    }
    // Free connection structures
    srv->RemoveConnection(conn->GetFd());

    // Notify about disconnection
    if (srv->client_disconnection_) {
      srv->client_disconnection_(*srv, *conn);
    }
  } else {
    MS_LOG(WARNING) << "Unhandled event:" << events
                    << " more detail see https://github.com/libevent/libevent/blob/master/include/event2/bufferevent.h";
  }
}

void TcpServer::SetTcpNoDelay(const evutil_socket_t &fd) {
  const int one = 1;
  int ret = setsockopt(fd, static_cast<int>(IPPROTO_TCP), static_cast<int>(TCP_NODELAY), &one, sizeof(int));
  if (ret < 0) {
    MS_LOG(EXCEPTION) << "Set socket no delay failed!";
  }
}

bool TcpServer::SendMessage(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const Protos &protos,
                            const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(data);
  return conn->SendMessage(meta, protos, data, size);
}

uint16_t TcpServer::BoundPort() const { return server_port_; }

std::string TcpServer::BoundIp() const { return server_address_; }

uint64_t TcpServer::ConnectionNum() const { return connections_.size(); }

const std::map<evutil_socket_t, std::shared_ptr<TcpConnection>> &TcpServer::Connections() const { return connections_; }

void TcpServer::SetMessageCallback(const OnServerReceiveMessage &cb) { message_callback_ = cb; }
}  // namespace fl
}  // namespace mindspore
