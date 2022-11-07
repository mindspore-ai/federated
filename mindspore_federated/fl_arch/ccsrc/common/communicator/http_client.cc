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

#include "common/communicator/http_client.h"

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
HttpClient::HttpClient(const std::string &remote_server_address)
    : remote_server_address_(std::move(remote_server_address)),
      event_base_(nullptr),
      buffer_event_(nullptr),
      http_req_(nullptr),
      evhttp_conn_(nullptr),
      uri_(nullptr),
      response_track_(nullptr) {}

HttpClient::~HttpClient() {
  if (buffer_event_) {
    bufferevent_free(buffer_event_);
    buffer_event_ = nullptr;
  }
  if (evhttp_conn_) {
    evhttp_connection_free(evhttp_conn_);
    evhttp_conn_ = nullptr;
  }
  if (event_base_) {
    event_base_free(event_base_);
    event_base_ = nullptr;
  }
}

void HttpClient::Init() {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  if (buffer_event_) {
    bufferevent_free(buffer_event_);
    buffer_event_ = nullptr;
  }
  if (!CommUtil::CheckHttpUrl(remote_server_address_)) {
    MS_LOG(EXCEPTION) << "The http client address:" << remote_server_address_ << " is illegal!";
  }

  int result = evthread_use_pthreads();
  if (result != 0) {
    MS_LOG(EXCEPTION) << "Use event pthread failed!";
  }

  if (event_base_ == nullptr) {
    event_base_ = event_base_new();
    MS_EXCEPTION_IF_NULL(event_base_);
  }
  if (!FLContext::instance()->enable_ssl()) {
    MS_LOG(INFO) << "SSL is disable.";
    buffer_event_ = bufferevent_socket_new(event_base_, -1, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  } else {
    if (!EstablishSSL()) {
      MS_LOG(EXCEPTION) << "Establish SSL failed.";
    }
  }
  MS_EXCEPTION_IF_NULL(buffer_event_);

  if (bufferevent_enable(buffer_event_, EV_READ | EV_WRITE) == -1) {
    MS_LOG(EXCEPTION) << "Buffer event enable read and write failed!";
  }

  uri_ = evhttp_uri_parse(remote_server_address_.c_str());
  int port = evhttp_uri_get_port(uri_);
  if (port == -1) {
    MS_LOG(EXCEPTION) << "Http uri port is invalid.";
  }

  if (evhttp_conn_) {
    evhttp_connection_free(evhttp_conn_);
    evhttp_conn_ = nullptr;
  }

  evhttp_conn_ =
    evhttp_connection_base_bufferevent_new(event_base_, nullptr, buffer_event_, evhttp_uri_get_host(uri_), port);
  MS_LOG(INFO) << "Host is:" << evhttp_uri_get_host(uri_) << ", port is:" << port;
}

bool HttpClient::Stop() {
  MS_ERROR_IF_NULL_W_RET_VAL(event_base_, false);
  std::lock_guard<std::mutex> lock(connection_mutex_);
  if (event_base_got_break(event_base_)) {
    MS_LOG(WARNING) << "The event base has already been stopped!";
    return false;
  }

  MS_LOG(INFO) << "Stop http client!";
  int ret = event_base_loopbreak(event_base_);
  if (ret != 0) {
    MS_LOG(ERROR) << "Event base loop break failed!";
    return false;
  }
  return true;
}

bool HttpClient::BreakLoopEvent() {
  MS_ERROR_IF_NULL_W_RET_VAL(event_base_, false);
  int ret = event_base_loopbreak(event_base_);
  if (ret != 0) {
    MS_LOG(ERROR) << "Event base loop break failed!";
    return false;
  }
  return true;
}

bool HttpClient::EstablishSSL() {
  MS_LOG(INFO) << "Enable http ssl support.";

  SSL *ssl = SSL_new(SSLClient::GetInstance().GetSSLCtx());
  MS_ERROR_IF_NULL_W_RET_VAL(ssl, false);
  MS_ERROR_IF_NULL_W_RET_VAL(event_base_, false);

  buffer_event_ = bufferevent_openssl_socket_new(event_base_, -1, ssl, BUFFEREVENT_SSL_CONNECTING,
                                                 BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  return true;
}

void HttpClient::ReadCallback(struct evhttp_request *http_req, void *const arg) {
  MS_ERROR_IF_NULL_WO_RET_VAL(http_req);
  MS_ERROR_IF_NULL_WO_RET_VAL(arg);
  auto http_client = reinterpret_cast<HttpClient *>(arg);
  MS_ERROR_IF_NULL_WO_RET_VAL(http_client);

  event_base *base = http_client->get_event_base();
  struct evkeyvalq *headers = evhttp_request_get_input_headers(http_req);
  auto rsp_message_id_ptr = evhttp_find_header(headers, "Message-Id");
  std::string rsp_message_id = rsp_message_id_ptr == nullptr ? "" : std::string(rsp_message_id_ptr);
  std::string expect_message_id = http_client->message_id();
  std::string target_msg_type = http_client->target_msg_type();
  const auto &response_track = http_client->response_track();

  if (rsp_message_id != http_client->message_id()) {
    MS_LOG(DEBUG) << "Response message id is different from the expect message id, rsp message id is " << rsp_message_id
                  << ", expect message id is " << expect_message_id;
    http_client->OnReadHandler(response_track, target_msg_type);
    event_base_loopbreak(base);
    return;
  }

  MS_ERROR_IF_NULL_WO_RET_VAL(base);
  MS_LOG(INFO) << "Response code is " << http_req->response_code << ", target msg type is " << target_msg_type
               << ", message id is " << rsp_message_id;
  switch (http_req->response_code) {
    case HTTP_OK: {
      struct evbuffer *evbuf = evhttp_request_get_input_buffer(http_req);
      size_t length = evbuffer_get_length(evbuf);
      MS_LOG(INFO) << "response message data length is:" << length;

      auto response_msg = std::make_shared<std::vector<uint8_t>>(length);
      int ret = memcpy_s(response_msg->data(), length, evbuffer_pullup(evbuf, -1), length);
      if (ret != 0) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return;
      }
      http_client->set_response_msg(response_msg);
      http_client->OnReadHandler(response_track, target_msg_type);
      event_base_loopbreak(base);
      break;
    }
    case HTTP_MOVEPERM:
      MS_LOG(WARNING) << "the uri moved permanently";
      break;
    default:
      MS_LOG(WARNING) << "Default: event base loop break.";
      event_base_loopbreak(base);
  }
}

void HttpClient::OnReadHandler(const std::shared_ptr<ResponseTrack> &response_track, const std::string msg_type) {
  if (message_callback_ != nullptr) {
    message_callback_(response_track, msg_type);
  }
}

void HttpClient::SetMessageCallback(const OnMessage &cb) { message_callback_ = cb; }

event_base *HttpClient::get_event_base() const { return event_base_; }

void HttpClient::set_response_track(const std::shared_ptr<ResponseTrack> &response_track) {
  response_track_ = response_track;
}

std::shared_ptr<ResponseTrack> HttpClient::response_track() const { return response_track_; }

void HttpClient::set_target_msg_type(const std::string target_msg_type) { target_msg_type_ = target_msg_type; }

std::string HttpClient::target_msg_type() const { return target_msg_type_; }

void HttpClient::set_message_id(const std::string message_id) { message_id_ = message_id; }

std::string HttpClient::message_id() const { return message_id_; }

void HttpClient::set_response_msg(const std::shared_ptr<std::vector<uint8_t>> &response_msg) {
  response_msg_ = response_msg;
}

const std::shared_ptr<std::vector<uint8_t>> HttpClient::response_msg() const { return response_msg_; }

bool HttpClient::SendMessage(const void *data, size_t data_size, const std::shared_ptr<ResponseTrack> &response_track,
                             const std::string &http_uri_path, const std::string &target_msg_type,
                             const std::string &message_source, const std::string &message_offset,
                             const std::string &content_type) {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  std::string message_id = CreateMessageId(response_track, target_msg_type, message_source);
  MS_LOG(INFO) << "target msg type is:" << target_msg_type << ", message source is " << message_source
               << ", data size is:" << data_size << ", request id:" << response_track->request_id()
               << ", remote server address is " << remote_server_address_ << ", message id is " << message_id;

  set_response_track(response_track);
  set_target_msg_type(target_msg_type);
  set_message_id(message_id);
  set_response_msg(nullptr);
  http_req_ = evhttp_request_new(ReadCallback, this);

  /** Set the post data */
  evbuffer_add(http_req_->output_buffer, data, data_size);
  evhttp_add_header(http_req_->output_headers, "Content-Type", content_type.c_str());
  evhttp_add_header(http_req_->output_headers, "Host", evhttp_uri_get_host(uri_));
  evhttp_add_header(http_req_->output_headers, "Message-Type", target_msg_type.c_str());
  evhttp_add_header(http_req_->output_headers, "Message-Source", message_source.c_str());
  evhttp_add_header(http_req_->output_headers, "Message-Id", message_id.c_str());
  evhttp_add_header(http_req_->output_headers, "Message-Offset", message_offset.c_str());
  evhttp_make_request(evhttp_conn_, http_req_, EVHTTP_REQ_POST, http_uri_path.c_str());

  int ret = event_base_dispatch(event_base_);
  if (ret != 0) {
    MS_LOG(ERROR) << "Event base dispatch failed!";
    return false;
  }
  return true;
}

bool HttpClient::SendMessage(const void *data, size_t data_size, const std::shared_ptr<ResponseTrack> &response_track,
                             const std::string &target_msg_type, const std::string &content_type) {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  MS_LOG(INFO) << "target msg type is:" << target_msg_type << ", data size is:" << data_size
               << ", request id:" << response_track->request_id() << ", remote server address is "
               << remote_server_address_;
  set_response_track(response_track);
  set_target_msg_type(target_msg_type);
  http_req_ = evhttp_request_new(ReadCallback, this);

  /** Set the post data */
  evbuffer_add(http_req_->output_buffer, data, data_size);
  evhttp_add_header(http_req_->output_headers, "Content-Type", content_type.c_str());
  evhttp_add_header(http_req_->output_headers, "Host", evhttp_uri_get_host(uri_));
  evhttp_make_request(evhttp_conn_, http_req_, EVHTTP_REQ_POST, target_msg_type.c_str());

  int ret = event_base_dispatch(event_base_);
  if (ret != 0) {
    MS_LOG(ERROR) << "Event base dispatch failed!";
    return false;
  }
  return true;
}

std::string HttpClient::CreateMessageId(const std::shared_ptr<ResponseTrack> &response_track,
                                        const std::string &target_msg_type, const std::string &message_source) {
  return message_source + ":" + target_msg_type + ":" + std::to_string(response_track->request_id());
}
}  // namespace fl
}  // namespace mindspore
