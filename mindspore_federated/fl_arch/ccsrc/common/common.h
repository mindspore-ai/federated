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

#ifndef MINDSPORE_CCSRC_FL_COMMON_COMMON_H_
#define MINDSPORE_CCSRC_FL_COMMON_COMMON_H_

#include <map>
#include <set>
#include <string>
#include <numeric>
#include <climits>
#include <memory>
#include <functional>
#include <iomanip>
#include <vector>
#include "common/protos/fl.pb.h"
#include "schema/fl_job_generated.h"
#include "schema/cipher_generated.h"
#include "common/communicator/http_message_handler.h"
#include "common/communicator/tcp_server.h"
#include "common/communicator/message_handler.h"
#include "common/status.h"

namespace mindspore {
namespace fl {
struct Address {
  Address() : addr(nullptr), size(0) {}
  Address(const void *address_addr, size_t address_size) : addr(address_addr), size(address_size) {}
  const void *addr;
  size_t size;
};

using AddressPtr = std::shared_ptr<Address>;

struct InputWeight {
  std::string name;
  const void *data = nullptr;
  size_t size = 0;
  std::vector<size_t> shape;
  std::string type;
  bool require_aggr = true;
};

struct WeightItem {
  std::string name;
  size_t offset = 0;
  size_t size = 0;
  std::vector<size_t> shape;
  std::string type;
  bool require_aggr = true;
};

struct ModelItem {
  std::vector<uint8_t> weight_data;
  size_t model_size = 0;
  std::map<std::string, WeightItem> weight_items;
  bool empty() const { return model_size == 0 || weight_data.empty() || weight_items.empty(); }
};

using ModelItemPtr = std::shared_ptr<ModelItem>;

enum TypeId : int {
  kTypeUnknown = 0,
  //
  // Meta types.
  //
  kNumberTypeFloat16,
  kNumberTypeFloat32,
  kNumberTypeFloat64,
  kNumberTypeUInt64
};

// Definitions for the server framework.
enum ServerMode { PARAMETER_SERVER = 0, FL_SERVER };
enum CommType { HTTP = 0, TCP };
enum AggregationType { FedAvg = 0, FedAdam, FedAdagarg, FedMeta, qffl, DenseGradAccum, SparseGradAccum };

struct RoundConfig {
  // The name of the round. Please refer to round kernel *.cc files.
  std::string name;
  // Whether this round has the time window limit.
  bool check_timeout = false;
  // The length of the time window. Only used when check_timeout is set to true.
  size_t time_window = 3000;
  // Whether this round has to check the request count has reach the threshold.
  bool check_count = false;
  // This round's request threshold count. Only used when threshold_count is set to true.
  size_t threshold_count = 0;
  // Use the total count of all valid servers as the final count, used for updateModel.
  bool per_server_count = false;
};

const std::set<std::string> DataRateKernels = {
  "startFLJob",   "updateModel", "getModel",      "exchangeKeys",       "getKeys",
  "shareSecrets", "getSecrets",  "getClientList", "reconstructSecrets", "pushListSign"};

using FBBuilder = flatbuffers::FlatBufferBuilder;
using TimeOutCb = std::function<void(bool, const std::string &)>;
using MessageCallback = std::function<void(const std::shared_ptr<MessageHandler> &)>;

// UploadData refers to the data which is uploaded by workers.
// Key refers to the data name. For example: "weights", "grad", "learning_rate", etc. This will be set by the worker.
// Value refers to the data of the key.

// We use Address instead of AddressPtr because:
// 1. Address doesn't need to call make_shared<T> so it has better performance.
// 2. The data uploaded by worker is normally parsed from FlatterBuffers or ProtoBuffer. For example: learning rate, new
// weights, etc. Address is enough to store these data.

// Pay attention that Address only stores the void* pointer of the data, so the data must not be released before the
// related logic is done.
using UploadData = std::map<std::string, Address>;

constexpr auto kWeight = "weight";
constexpr auto kNewWeight = "new_weight";
constexpr auto kStartFLJobTotalClientNum = "startFLJobTotalClientNum";
constexpr auto kStartFLJobAcceptClientNum = "startFLJobAcceptClientNum";
constexpr auto kStartFLJobRejectClientNum = "startFLJobRejectClientNum";
constexpr auto kUpdateModelTotalClientNum = "updateModelTotalClientNum";
constexpr auto kUpdateModelAcceptClientNum = "updateModelAcceptClientNum";
constexpr auto kUpdateModelRejectClientNum = "updateModelRejectClientNum";
constexpr auto kGetModelTotalClientNum = "getModelTotalClientNum";
constexpr auto kGetModelAcceptClientNum = "getModelAcceptClientNum";
constexpr auto kGetModelRejectClientNum = "getModelRejectClientNum";
constexpr auto kParticipationTimeLevel1 = "participationTimeLevel1";
constexpr auto kParticipationTimeLevel2 = "participationTimeLevel2";
constexpr auto kParticipationTimeLevel3 = "participationTimeLevel3";
constexpr auto kMinVal = "min_val";
constexpr auto kMaxVal = "max_val";
constexpr auto kDataSize = "data_size";
constexpr auto kNewDataSize = "new_data_size";

constexpr uint32_t kLeaderServerRank = 0;
constexpr size_t kWorkerMgrThreadPoolSize = 32;
constexpr size_t kWorkerMgrMaxTaskNum = 64;
constexpr size_t kCipherMgrThreadPoolSize = 32;
constexpr size_t kCipherMgrMaxTaskNum = 64;
constexpr size_t kExecutorThreadPoolSize = 32;
constexpr size_t kExecutorMaxTaskNum = 32;
constexpr size_t kNumberTypeFloat16Type = 2;
constexpr size_t kNumberTypeFloat32Type = 4;
constexpr size_t kNumberTypeUInt64Type = 8;
constexpr uint32_t kThreadSleepTime = 50;
constexpr auto kPBProtocol = "PB";
constexpr auto kFBSProtocol = "FBS";
constexpr auto kFedAvg = "FedAvg";
constexpr auto kAggregationKernelType = "Aggregation";
constexpr auto kCtxIterNum = "iteration";
constexpr auto kCtxDeviceMetas = "device_metas";
constexpr auto kCtxTotalTimeoutDuration = "total_timeout_duration";
constexpr auto kCtxIterationNextRequestTimestamp = "iteration_next_request_timestamp";
constexpr auto kCtxUpdateModelThld = "update_model_threshold";
constexpr auto kCtxFedAvgTotalDataSize = "fed_avg_total_data_size";
constexpr auto kCurrentIteration = "current_iteration";
constexpr auto kInstanceState = "instance_state";
constexpr auto SECRET_MAX_LEN = 32;
constexpr auto PRIME_MAX_LEN = 33;
const char PYTHON_MOD_SERIALIZE_MODULE[] = "mindspore.train.serialization";
const char PYTHON_MOD_SAFE_WEIGHT[] = "_save_weight";

// This macro the current timestamp in milliseconds.
#define CURRENT_TIME_MILLI \
  std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())

// Definitions for Federated Learning.

constexpr auto kNetworkError = "Cluster networking failed.";
constexpr auto KTriggerCounterEventError = "Cluster trigger counter event failed.";

// The result code used for round kernels.
enum class ResultCode {
  // If the method is successfully called and round kernel's residual methods should be called, return kSuccess.
  kSuccess = 0,
  // If there's error happened, return kFail.
  kFail
};

inline std::string GetEnv(const std::string &env_var) {
  const char *value = ::getenv(env_var.c_str());
  if (value == nullptr) {
    return std::string();
  }
  return std::string(value);
}

inline bool isNaN(float num) { return num != num; }
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMMON_COMMON_H_
