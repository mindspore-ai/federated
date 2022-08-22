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

#ifndef MINDSPORE_FL_ARCH_CCSRC_VERTICAL_TRAINER_COMMUNICATOR_H_
#define MINDSPORE_FL_ARCH_CCSRC_VERTICAL_TRAINER_COMMUNICATOR_H_

#include <utility>
#include <string>
#include <vector>
#include <memory>

#include "vertical/communicator/abstract_communicator.h"
#include "vertical/common.h"
#include "common/communicator/http_client.h"
#include "common/protos/vfl.pb.h"
#include "vertical/python/tensor_list_py.h"
#include "vertical/python/tensor_py.h"
#include "vertical/communicator/message_queue.h"

namespace mindspore {
namespace fl {
class TrainerCommunicator : public AbstractCommunicator {
 public:
  TrainerCommunicator() = default;
  ~TrainerCommunicator() = default;

  static TrainerCommunicator &GetInstance();

  void InitHttpClient();

  bool LaunchMsgHandler(const std::shared_ptr<MessageHandler> &message) override;

  std::string Send(const void *data, size_t data_size, const std::string &msg_type, const std::string &content_type);

  bool Start(const uint32_t &timeout = 1000) override;

  bool Stop() override;

  bool Send(const TensorListItemPy &tensorListItemPy);

  TensorListItemPy Receive(const uint32_t &timeout = 100000);

 private:
  bool VerifyTensorListProto(const TensorListProto &start_fl_job_req);

  TensorListItemPy ParseTensorListProto(const TensorListProto &tensorListProto);

  void CreateTensorProto(TensorProto *tensor_proto, const TensorItemPy &tensor, std::string ref_key);

  void CreateTensorListProto(TensorListProto *tensor_list_proto, const TensorListItemPy &tensorListItemPy);

  bool TensorMsgReceiveHandler(const TensorListItemPy &tensorListItemPy);

  void InitTrainerCommunicator(const std::shared_ptr<CommunicatorBase> &http_communicator);

  void InitTrainerConfigs();

  std::string remote_server_address_;

  std::shared_ptr<HttpClient> http_client_ = nullptr;

  std::vector<TrainerConfig> trainer_config_ = {};

  bool is_message_received;

  std::mutex message_received_mutex_;

  std::condition_variable message_received_cond_;

  bool is_message_received_;

  std::shared_ptr<MessageQueue<TensorListItemPy>> message_queue_ = nullptr;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_TRAINER_COMMUNICATOR_H_
