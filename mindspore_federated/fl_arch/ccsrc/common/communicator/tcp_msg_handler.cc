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

#include "communicator/tcp_msg_handler.h"
#include <memory>
#include <string>

namespace mindspore {
namespace fl {
TcpMsgHandler::TcpMsgHandler(const std::shared_ptr<TcpConnection> &conn, const MessageMeta &meta, const VectorPtr &data)
    : tcp_conn_(conn), meta_(meta), data_(data) {}

const void *TcpMsgHandler::data() const {
  if (data_ == nullptr) {
    return nullptr;
  }
  return data_->data();
}

size_t TcpMsgHandler::len() const {
  if (data_ == nullptr) {
    return 0;
  }
  return data_->size();
}

bool TcpMsgHandler::SendResponse(const void *data, const size_t &len) {
  MS_ERROR_IF_NULL_W_RET_VAL(tcp_conn_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(data, false);

  MS_LOG(DEBUG) << "Response tcp message, this node id:" << meta_.recv_node()
                << ", request node id: " << meta_.send_node() << ", request id:" << meta_.request_id();
  if (!tcp_conn_->SendMessage(meta_, Protos::RAW, data, len)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
  has_sent_response_ = true;
  return true;
}

std::string TcpMsgHandler::message_type() const { return message_type_; }
}  // namespace fl
}  // namespace mindspore
