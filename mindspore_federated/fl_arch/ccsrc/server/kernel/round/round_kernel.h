/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_ROUND_ROUND_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_ROUND_ROUND_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <queue>
#include <utility>
#include <chrono>
#include <thread>
#include <unordered_map>
#include "common/common.h"
#include "server/local_meta_store.h"
#include "server/distributed_count_service.h"
#include "distributed_cache/client_infos.h"
#include "distributed_cache/counter.h"
#include "distributed_cache/instance_context.h"
#include "server/executor.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
// results of signature verification
enum sigVerifyResult { FAILED, TIMEOUT, PASSED };

constexpr uint64_t kReleaseDuration = 100;
// RoundKernel contains the main logic of server handling messages from workers. One iteration has multiple round
// kernels to represent the process. They receive and parse messages from the server communication module. After
// handling these messages, round kernels allocate response data and send it back.

// For example, the main process of federated learning is:
// startFLJob round->updateModel round->getModel round.
class RoundKernel {
 public:
  RoundKernel();
  virtual ~RoundKernel();

  size_t iteration_time_window() const { return iteration_time_window_; }
  void InitKernelCommon(size_t iteration_time_window);
  // Initialize RoundKernel with threshold_count which means that for every iteration, this round needs threshold_count
  // messages.
  virtual void InitKernel(size_t threshold_count) = 0;

  // Launch the round kernel logic to handle the message passed by the communication module.
  virtual bool Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) = 0;

  // Some rounds could be stateful in a iteration. Reset method resets the status of this round.
  virtual bool Reset() { return true; }

  // The counter event handlers for DistributedCountService.
  // The callbacks when first message and last message for this round kernel is received.
  // These methods is called by class DistributedCountService and triggered by counting server.
  virtual void OnFirstCountEvent();
  virtual void OnLastCountEvent();

  // Set round kernel name, which could be used in round kernel's methods.
  void set_name(const std::string &name);

  void Summarize();

  void IncreaseTotalClientNum();

  void IncreaseAcceptClientNum();

  size_t total_client_num() const;

  size_t accept_client_num() const;

  size_t reject_client_num() const;

  void InitClientVisitedNum();

  void InitClientUploadLoss();

  void UpdateClientUploadLoss(const float upload_loss);

  float upload_loss() const;

  bool verifyResponse(const std::shared_ptr<MessageHandler> &message, const void *data, size_t len);

  // Record the size of send data and the time stamp
  void RecordSendData(const std::pair<uint64_t, size_t> &send_data);

  // Record the size of receive data and the time stamp
  void RecordReceiveData(const std::pair<uint64_t, size_t> &receive_data);

  // Get the info of send data
  std::multimap<uint64_t, size_t> GetSendData();

  // Get the info of receive data
  std::multimap<uint64_t, size_t> GetReceiveData();

  // Clear the send data infp
  void ClearData();

 protected:
  // Send response to client, and the data can be released after the call.
  void SendResponseMsg(const std::shared_ptr<MessageHandler> &message, const void *data, size_t len);
  // Send response to client, and the data will be released by cb after finished send msg.
  void SendResponseMsgInference(const std::shared_ptr<MessageHandler> &message, const void *data, size_t len,
                                RefBufferRelCallback cb);
  sigVerifyResult VerifySignatureBase(const std::string &fl_id, const std::vector<std::string> &src_data,
                                      const flatbuffers::Vector<uint8_t> *signature, const std::string &timestamp);
  sigVerifyResult VerifySignatureBase(const std::string &fl_id, const std::vector<uint8_t> &src_data,
                                      const flatbuffers::Vector<uint8_t> *signature, const std::string &timestamp);

  template <class T>
  sigVerifyResult VerifySignatureBase(const T *request) {
    MS_ERROR_IF_NULL_W_RET_VAL(request, sigVerifyResult::FAILED);
    MS_ERROR_IF_NULL_W_RET_VAL(request->fl_id(), sigVerifyResult::FAILED);
    MS_ERROR_IF_NULL_W_RET_VAL(request->timestamp(), sigVerifyResult::FAILED);

    std::string fl_id = request->fl_id()->str();
    std::string timestamp = request->timestamp()->str();
    int iteration = request->iteration();
    std::string iter_str = std::to_string(iteration);
    return VerifySignatureBase(fl_id, {timestamp, iter_str}, request->signature(), timestamp);
  }
  // Round kernel's name.
  std::string name_;

  std::atomic<size_t> total_client_num_ = 0;
  std::atomic<size_t> accept_client_num_ = 0;

  std::atomic<float> upload_loss_ = 0.0;
  size_t iteration_time_window_ = 0;
  Executor *executor_ = nullptr;

  // The mutex for send_data_and_time_
  std::mutex send_data_rate_mutex_;

  // The size of send data ant time
  std::multimap<uint64_t, size_t> send_data_and_time_;

  // The mutex for receive_data_and_time_
  std::mutex receive_data_rate_mutex_;

  // The size of receive data and time
  std::multimap<uint64_t, size_t> receive_data_and_time_;
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_ROUND_ROUND_KERNEL_H_
