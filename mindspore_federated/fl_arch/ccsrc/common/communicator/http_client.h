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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_HTTP_CLIENT_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_HTTP_CLIENT_H_

#include <event2/event.h>
#include <event2/bufferevent.h>
#include <event2/thread.h>
#include <event2/bufferevent_ssl.h>
#include <event2/buffer.h>
#include <event2/http.h>
#include <event2/http_struct.h>
#include <event2/keyvalq_struct.h>

#include <functional>
#include <string>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "common/core/comm_util.h"
#include "utils/convert_utils_base.h"
#include "common/communicator/ssl_client.h"
#include "common/core/abstract_node.h"

namespace mindspore {
namespace fl {
#define HTTP_CONTENT_TYPE_URL_ENCODED "application/x-www-form-urlencoded"
#define HTTP_CONTENT_TYPE_FORM_DATA "multipart/form-data"
#define HTTP_CONTENT_TYPE_TEXT_PLAIN "text/plain"

class HttpClient {
 public:
  using OnConnected = std::function<void()>;
  using OnDisconnected = std::function<void()>;
  using OnTimeout = std::function<void()>;
  using OnMessage =
    std::function<void(const std::shared_ptr<ResponseTrack> &response_track, const std::string &msg_type)>;
  using OnTimer = std::function<void()>;

  explicit HttpClient(const std::string &http_server_address);
  virtual ~HttpClient();

  void set_disconnected_callback(const OnDisconnected &disconnected);
  void set_connected_callback(const OnConnected &connected);
  void Init();
  bool Stop();
  void SetMessageCallback(const OnMessage &cb);
  bool SendMessage(const void *data, size_t data_size, const std::shared_ptr<ResponseTrack> &response_track,
                   const std::string &target_msg_type, const std::string &content_type);
  bool SendMessage(const void *data, size_t data_size, const std::shared_ptr<ResponseTrack> &response_track,
                   const std::string &target_msg_type, const std::string &request_msg_type,
                   const std::string &content_type);
  event_base *get_event_base() const;
  bool BreakLoopEvent();
  void set_response_track(const std::shared_ptr<ResponseTrack> &response_track);
  std::shared_ptr<ResponseTrack> response_track() const;
  void set_target_msg_type(const std::string target_msg_type);
  std::string target_msg_type() const;
  void set_response_msg(const std::shared_ptr<std::vector<unsigned char>> &response_msg);
  const std::shared_ptr<std::vector<unsigned char>> response_msg() const;

 protected:
  static void ReadCallback(struct evhttp_request *http_req, void *message_callback);
  bool EstablishSSL();
  void OnReadHandler(const std::shared_ptr<ResponseTrack> &response_track, const std::string kernel_name);
  std::string PeerRoleName() const;

 private:
  OnMessage message_callback_;

  OnConnected connected_callback_;
  OnDisconnected disconnected_callback_;
  OnTimeout timeout_callback_;
  OnTimer on_timer_callback_;

  std::string remote_server_address_;
  std::string target_msg_type_;
  event_base *event_base_;
  bufferevent *buffer_event_;

  std::mutex connection_mutex_;
  std::condition_variable connection_cond_;

  std::uint16_t server_port_;
  evhttp_request *http_req_;
  evhttp_connection *evhttp_conn_;
  evhttp_uri *uri;
  std::shared_ptr<ResponseTrack> response_track_;
  std::shared_ptr<std::vector<unsigned char>> response_msg_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_HTTP_CLIENT_H_
