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

#include "common/communicator/tcp_client.h"

#include <arpa/inet.h>
#include <event2/buffer.h>
#include <event2/buffer_compat.h>
#include <event2/bufferevent.h>
#include <event2/event.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>

namespace mindspore {
namespace fl {
bufferevent *TcpClientDispatch::CreateBufferEvent() {
  std::unique_lock<std::mutex> lock(event_base_mutex_);
  int result = evthread_use_pthreads();
  if (result != 0) {
    MS_LOG(ERROR) << "Use event pthread failed!";
    return nullptr;
  }
  if (event_base_ == nullptr) {
    event_base_ = event_base_new();
    if (event_base_ == nullptr) {
      MS_LOG(ERROR) << "Call event_base_new failed!";
      return nullptr;
    }
  }
  bufferevent *buffer_event;
  if (!FLContext::instance()->enable_ssl()) {
    MS_LOG(INFO) << "SSL is disable.";
    buffer_event = bufferevent_socket_new(event_base_, -1, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  } else {
    MS_LOG(INFO) << "Enable ssl support.";
    SSL *ssl = SSL_new(SSLClient::GetInstance().GetSSLCtx());
    if (ssl == nullptr) {
      MS_LOG_WARNING << "Call SSL_new failed";
      return nullptr;
    }
    buffer_event = bufferevent_openssl_socket_new(event_base_, -1, ssl, BUFFEREVENT_SSL_CONNECTING,
                                                  BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  }
  if (buffer_event == nullptr) {
    MS_LOG_WARNING << "Create buffer event failed";
    return nullptr;
  }
  return buffer_event;
}

void TcpClientDispatch::Stop() {
  std::lock_guard<std::mutex> lock(event_base_mutex_);
  MS_LOG(INFO) << "Stop tcp client event dispatch!";
  if (event_base_ == nullptr) {
    return;
  }
  if (is_started_) {
    int ret = event_base_loopbreak(event_base_);
    if (ret != 0) {
      MS_LOG(ERROR) << "Event base loop break failed!";
    }
  }
  if (dispatch_thread_.joinable()) {
    dispatch_thread_.join();
  }
  event_base_free(event_base_);
  event_base_ = nullptr;
}

void TcpClientDispatch::StartDispatch() {
  std::lock_guard<std::mutex> lock(event_base_mutex_);
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

TcpClient::TcpClient(const std::string &address, std::uint16_t port, NodeRole peer_role)
    : buffer_event_(nullptr), server_address_(std::move(address)), server_port_(port), peer_role_(peer_role) {}

TcpClient::~TcpClient() { Stop(); }

std::string TcpClient::GetServerAddress() const { return server_address_; }

void TcpClient::set_disconnected_callback(const OnDisconnected &disconnected) { disconnected_callback_ = disconnected; }

void TcpClient::set_connected_callback(const OnConnected &connected) { connected_callback_ = connected; }

std::string TcpClient::PeerRoleName() const {
  switch (peer_role_) {
    case SERVER:
      return "Server";
    case WORKER:
      return "Worker";
    case SCHEDULER:
      return "Scheduler";
    default:
      return "RoleUndefined";
  }
}

bool TcpClient::WaitConnected(const uint32_t &connected_timeout) {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  (void)connection_cond_.wait_for(lock, std::chrono::seconds(connected_timeout), [this] { return connected_.load(); });
  return connected_;
}

void TcpClient::Stop() {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  MS_LOG(INFO) << "Stop tcp client!";
  if (buffer_event_) {
    bufferevent_free(buffer_event_);
    buffer_event_ = nullptr;
  }
  connected_ = false;
}

void TcpClient::SetTcpNoDelay(const evutil_socket_t &fd) {
  const int one = 1;
  int ret = setsockopt(fd, static_cast<int>(IPPROTO_TCP), static_cast<int>(TCP_NODELAY), &one, sizeof(int));
  if (ret < 0) {
    MS_LOG(EXCEPTION) << "Set socket no delay failed!";
  }
}

void TcpClient::ReadCallback(struct bufferevent *bev, void *const ctx) {
  try {
    MS_EXCEPTION_IF_NULL(ctx);
    auto tcp_client = reinterpret_cast<TcpClient *>(ctx);
    MS_EXCEPTION_IF_NULL(tcp_client);
    auto read_fun = [bev](void *data, size_t max_size) { return bufferevent_read(bev, data, max_size); };
    tcp_client->message_handler_.ReceiveMessage(read_fun);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpClient::NotifyConnected() {
  MS_LOG(INFO) << "Client connected to the server! Peer " << PeerRoleName() << " ip: " << server_address_
               << ", port: " << server_port_;
  connected_ = true;
  connection_cond_.notify_all();
}

void TcpClient::NotifyNotConnected() {
  MS_LOG(INFO) << "Client failed to connect to the server! Peer " << PeerRoleName() << " ip: " << server_address_
               << ", port: " << server_port_;
  connected_ = false;
  connection_cond_.notify_all();
}

void TcpClient::EventCallback(struct bufferevent *bev, std::int16_t events, void *const ptr) {
  try {
    MS_EXCEPTION_IF_NULL(ptr);
    auto tcp_client = reinterpret_cast<TcpClient *>(ptr);
    tcp_client->EventCallbackInner(bev, events);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpClient::EventCallbackInner(struct bufferevent *bev, std::int16_t events) {
  MS_EXCEPTION_IF_NULL(bev);
  if (events & BEV_EVENT_CONNECTED) {
    // Connected
    if (connected_callback_) {
      connected_callback_();
    }
    NotifyConnected();
    evutil_socket_t fd = bufferevent_getfd(bev);
    SetTcpNoDelay(fd);
    MS_LOG(INFO) << "Client connected! Peer " << PeerRoleName() << " ip: " << server_address_
                 << ", port: " << server_port_;
  } else if (events & BEV_EVENT_ERROR) {
    MS_LOG(WARNING) << "BEV_EVENT_ERROR event is trigger!";
    if (FLContext::instance()->enable_ssl()) {
      uint64_t err = bufferevent_get_openssl_error(bev);

      MS_LOG(DEBUG) << "The error number is:" << err
                      <<"Error message:" << ERR_reason_error_string(err)
                      << ", the error lib:" << ERR_lib_error_string(err)
                      << ", the error func:" << ERR_func_error_string(err);
      MS_LOG(ERROR) << "Tcp client connect filed!";
    }
    connected_ = false;
    if (disconnected_callback_) {
      disconnected_callback_();
    }
  } else if (events & BEV_EVENT_EOF) {
    MS_LOG(WARNING) << "Client connected end of file! Peer " << PeerRoleName() << " ip: " << server_address_
                    << ", port: " << server_port_;
    connected_ = false;
    if (disconnected_callback_) {
      disconnected_callback_();
    }
  }
}

bool TcpClient::Start(uint64_t timeout_in_seconds) {
  if (!CommUtil::CheckIp(server_address_)) {
    MS_LOG(WARNING) << "The tcp client ip:" << server_address_ << " is illegal!";
    return false;
  }
  auto buffer_event = TcpClientDispatch::Instance().CreateBufferEvent();
  if (buffer_event == nullptr) {
    MS_LOG(WARNING) << "Create buffer event for tcp client failed";
    return false;
  }
  bufferevent_setcb(buffer_event, ReadCallback, nullptr, EventCallback, this);
  if (bufferevent_enable(buffer_event, EV_READ | EV_WRITE) == -1) {
    MS_LOG_WARNING << "Buffer event enable read and write failed!";
    bufferevent_free(buffer_event);
    return false;
  }
  sockaddr_in sin{};
  if (memset_s(&sin, sizeof(sin), 0, sizeof(sin)) != EOK) {
    MS_LOG(WARNING) << "Initialize sockaddr_in failed!";
  }
  sin.sin_family = AF_INET;
  sin.sin_addr.s_addr = inet_addr(server_address_.c_str());
  sin.sin_port = htons(server_port_);
  int result_code = bufferevent_socket_connect(buffer_event, reinterpret_cast<struct sockaddr *>(&sin), sizeof(sin));
  if (result_code < 0) {
    MS_LOG(WARNING) << "Connect server ip:" << server_address_ << " and port: " << server_port_ << " is failed!";
    bufferevent_free(buffer_event);
    return false;
  }
  TcpClientDispatch::Instance().StartDispatch();
  auto res = WaitConnected(timeout_in_seconds);
  if (!res) {
    MS_LOG(WARNING) << "Connect to server ip:" << server_address_ << " and port: " << server_port_ << " is failed!";
    bufferevent_free(buffer_event);
    return false;
  }
  buffer_event_ = buffer_event;
  return true;
}

void TcpClient::SetMessageCallback(const TcpMessageHandler::MessageHandleFun &cb) { message_handler_.SetCallback(cb); }

bool TcpClient::SendMessage(const MessageMeta &meta, const Protos &protos, const void *data, size_t size) {
  if (buffer_event_ == nullptr) {
    MS_LOG(ERROR) << "Event buffer not inited!";
    return false;
  }
  if (data == nullptr) {
    MS_LOG(ERROR) << "Input data cannot be nullptr!";
    return false;
  }
  bufferevent_lock(buffer_event_);
  bool res = true;

  const std::string &meta_str = meta.SerializeAsString();
  MessageHeader header;
  header.message_proto_ = protos;
  header.message_meta_length_ = SizeToUint(meta_str.size());
  header.message_length_ = size + header.message_meta_length_;

  if (bufferevent_write(buffer_event_, &header, sizeof(header)) == -1) {
    MS_LOG(ERROR) << "Event buffer add header failed!";
    res = false;
  }
  if (bufferevent_write(buffer_event_, meta_str.data(), meta_str.size()) == -1) {
    MS_LOG(ERROR) << "Event buffer add protobuf data failed!";
    res = false;
  }
  if (bufferevent_write(buffer_event_, data, size) == -1) {
    MS_LOG(ERROR) << "Event buffer add protobuf data failed!";
    res = false;
  }
  int result = bufferevent_flush(buffer_event_, EV_READ | EV_WRITE, BEV_FLUSH);
  if (result < 0) {
    MS_LOG(ERROR) << "Bufferevent flush failed!";
    res = false;
  }
  bufferevent_unlock(buffer_event_);
  return res;
}
}  // namespace fl
}  // namespace mindspore
