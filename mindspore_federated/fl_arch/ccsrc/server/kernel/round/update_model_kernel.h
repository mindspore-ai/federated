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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_UPDATE_MODEL_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_UPDATE_MODEL_KERNEL_H_

#include <map>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include "common/common.h"
#include "server/kernel/round/round_kernel.h"
#include "server/kernel/round/round_kernel_factory.h"
#include "server/executor.h"
#include "server/model_store.h"
#include "armour/cipher/cipher_meta_storage.h"
#include "compression/decode_executor.h"
#include "schema/fl_job_generated.h"
#include "schema/cipher_generated.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
// The initial data size sum of federated learning is 0, which will be accumulated in updateModel round.
constexpr uint64_t kInitialDataSizeSum = 0;
class UpdateModelKernel : public RoundKernel {
 public:
  UpdateModelKernel() = default;
  ~UpdateModelKernel() override = default;

  void InitKernel(size_t threshold_count) override;
  bool Launch(const uint8_t *req_data, size_t len, const std::shared_ptr<MessageHandler> &message) override;
  bool Reset() override;

  // In some cases, the last updateModel message means this server iteration is finished.
  void OnLastCountEvent() override;

  // Get participation_time_and_num_
  const std::vector<std::pair<uint64_t, uint32_t>> &GetCompletePeriodRecord();

  // Reset participation_time_and_num_
  void ResetParticipationTimeAndNum();

 private:
  ResultCode ReachThresholdForUpdateModel(const std::shared_ptr<FBBuilder> &fbb,
                                          const schema::RequestUpdateModel *update_model_req);
  ResultCode UpdateModel(const schema::RequestUpdateModel *update_model_req, const std::shared_ptr<FBBuilder> &fbb,
                         const DeviceMeta &device_meta, const std::map<std::string, Address> &feature_map);
  ResultCode ParseAndVerifyFeatureMap(const schema::RequestUpdateModel *update_model_req, const DeviceMeta &device_meta,
                                      const std::shared_ptr<FBBuilder> &fbb,
                                      std::map<std::string, std::vector<float>> *weight_map_ptr,
                                      std::map<std::string, Address> *feature_map_ptr);

  std::map<std::string, Address> ParseFeatureMap(const schema::RequestUpdateModel *update_model_req);
  std::map<std::string, Address> ParseSignDSFeatureMap(const schema::RequestUpdateModel *update_model_req,
                                                       size_t data_size,
                                                       std::map<std::string, std::vector<float>> *weight_map);
  std::map<std::string, Address> ParseUploadCompressFeatureMap(const schema::RequestUpdateModel *update_model_req,
                                                               size_t data_size,
                                                               std::map<std::string, std::vector<float>> *weight_map);
  bool VerifySignDSFeatureMap(const std::unordered_map<std::string, size_t> &model,
                              const schema::RequestUpdateModel *update_model_req);
  bool VerifyUploadCompressFeatureMap(const schema::RequestUpdateModel *update_model_req);
  sigVerifyResult VerifySignature(const schema::RequestUpdateModel *update_model_req);
  void BuildUpdateModelRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                           const std::string &reason, const std::string &next_req_time);
  ResultCode VerifyUpdateModel(const schema::RequestUpdateModel *update_model_req,
                               const std::shared_ptr<FBBuilder> &fbb, DeviceMeta *device_meta);

  // Decode functions of compression.
  std::map<std::string, Address> DecodeFeatureMap(std::map<std::string, std::vector<float>> *weight_map,
                                                  const schema::RequestUpdateModel *update_model_req,
                                                  schema::CompressType upload_compress_type, size_t data_size);
  // Record complete update model number according to participation_time_level
  void RecordCompletePeriod(const DeviceMeta &device_meta);

  // Check and transform participation time level parament
  void CheckAndTransPara(const std::string &participation_time_level);

  bool VerifyUpdateModelRequest(const schema::RequestUpdateModel *update_model_req);

  // Check upload mode
  bool IsCompress(const schema::RequestUpdateModel *update_model_req);

  // From StartFlJob to UpdateModel complete time and number
  std::vector<std::pair<uint64_t, uint32_t>> participation_time_and_num_{};

  // The mutex for participation_time_and_num_
  std::mutex participation_time_and_num_mtx_;
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_UPDATE_MODEL_KERNEL_H_
