/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "communicator/http_communicator.h"
#include <memory>

namespace mindspore {
namespace fl {
void HttpCommunicator::RegisterRoundMsgCallback(const std::string &msg_type, const MessageCallback &callback) {
  MS_LOG(INFO) << "msg_type is: " << msg_type;
  http_msg_callbacks_[msg_type] = [msg_type, callback](const std::shared_ptr<HttpMessageHandler> &http_msg) -> void {
    MS_EXCEPTION_IF_NULL(http_msg);
    try {
      size_t len = 0;
      void *data = nullptr;
      if (!http_msg->GetPostMsg(&len, &data)) {
        FlStatus result(kInvalidInputs, "Get post message failed");
        http_msg->ErrorResponse(HTTP_INTERNAL, result);
        return;
      }
      std::string message_type = http_msg->GetHeadParam("Message-Type");
      std::string message_id = http_msg->GetHeadParam("Message-Id");
      std::string message_source = http_msg->GetHeadParam("Message-Source");
      std::shared_ptr<MessageHandler> http_msg_handler =
        std::make_shared<HttpMsgHandler>(http_msg, data, len, message_type, message_id, message_source);
      MS_EXCEPTION_IF_NULL(http_msg_handler);
      callback(http_msg_handler);
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Catch exception when invoke message handler, msg_type: " << msg_type
                    << " exception: " << e.what();
      FlStatus result(kSystemError, e.what());
      http_msg->ErrorResponse(HTTP_INTERNAL, result);
    }
  };

  std::string url = FLContext::instance()->http_url_prefix();
  if (url.empty()) {
    url += "/";
  }
  url += msg_type;
  MS_EXCEPTION_IF_NULL(http_server_);
  bool is_succeed = http_server_->RegisterRoute(url, &http_msg_callbacks_[msg_type]);
  if (!is_succeed) {
    MS_LOG(EXCEPTION) << "Http server register handler for url " << url << " failed.";
  }
}

std::shared_ptr<HttpServer> &HttpCommunicator::http_server() { return http_server_; }
}  // namespace fl
}  // namespace mindspore
