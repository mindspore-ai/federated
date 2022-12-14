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

#include "server/kernel/round/reconstruct_secrets_kernel.h"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <utility>

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void ReconstructSecretsKernel::InitKernel(size_t) {
  name_unmask_ = "UnMaskKernel";
  MS_LOG(INFO) << "ReconstructSecretsKernel Load, ITERATION NUMBER IS : "
               << cache::InstanceContext::Instance().iteration_num();
}

sigVerifyResult ReconstructSecretsKernel::VerifySignature(const schema::SendReconstructSecret *reconstruct_secret_req) {
  return VerifySignatureBase(reconstruct_secret_req);
}

bool ReconstructSecretsKernel::Launch(const uint8_t *req_data, size_t len,
                                      const std::shared_ptr<MessageHandler> &message) {
  bool response = false;
  size_t iter_num = cache::InstanceContext::Instance().iteration_num();
  MS_LOG(INFO) << "Launching ReconstructSecrets Kernel, Iteration number is " << iter_num;

  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    return false;
  }

  // get client list from memory server.
  std::vector<std::string> update_model_clients;
  auto status = fl::cache::ClientInfos::GetInstance().GetAllUpdateModelClients(&update_model_clients);
  if (!status.IsSuccess()) {
    std::string reason = "Get update model client list failed.";
    cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_RequestError, reason, SizeToInt(iter_num),
                                                   std::to_string(CURRENT_TIME_MILLI.count()));
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }
  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::SendReconstructSecret>()) {
    std::string reason = "The schema of SendReconstructSecret is invalid.";
    cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_RequestError, reason, SizeToInt(iter_num),
                                                   std::to_string(CURRENT_TIME_MILLI.count()));
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  const schema::SendReconstructSecret *reconstruct_secret_req =
    flatbuffers::GetRoot<schema::SendReconstructSecret>(req_data);
  if (reconstruct_secret_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for SendReconstructSecret.";
    cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_RequestError, reason, SizeToInt(iter_num),
                                                   std::to_string(CURRENT_TIME_MILLI.count()));
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  // verify signature
  if (FLContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(reconstruct_secret_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed.";
      cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_RequestError, reason,
                                                     SizeToInt(iter_num), std::to_string(CURRENT_TIME_MILLI.count()));
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed or cannot find its key attestation.";
      cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_OutOfTime, reason, SizeToInt(iter_num),
                                                     std::to_string(CURRENT_TIME_MILLI.count()));
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }
    MS_LOG(INFO) << "verify signature passed!";
  }

  std::string fl_id = reconstruct_secret_req->fl_id()->str();
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(ERROR) << "Current amount for ReconstructSecretsKernel is enough.";
    if (find(update_model_clients.begin(), update_model_clients.end(), fl_id) != update_model_clients.end()) {
      // client in get update model client list.
      cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_SUCCEED,
                                                     "Current amount for ReconstructSecretsKernel is enough.",
                                                     SizeToInt(iter_num), std::to_string(CURRENT_TIME_MILLI.count()));
    } else {
      cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_OutOfTime,
                                                     "Current amount for ReconstructSecretsKernel is enough.",
                                                     SizeToInt(iter_num), std::to_string(CURRENT_TIME_MILLI.count()));
    }
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  auto cur_iterator = SizeToInt(iter_num);
  auto next_req_time = std::to_string(CURRENT_TIME_MILLI.count());
  response = cipher_reconstruct_.ReconstructSecrets(cur_iterator, next_req_time, reconstruct_secret_req, fbb,
                                                    update_model_clients);

  if (response) {
    (void)DistributedCountService::GetInstance().Count(name_);
  }

  if (!checkReachThreshold(update_model_clients, cur_iterator, next_req_time, fbb, fl_id)) {
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());

  MS_LOG(INFO) << "reconstruct_secrets_kernel success.";
  if (!response) {
    MS_LOG(INFO) << "reconstruct_secrets_kernel response not ready.";
  }
  return true;
}

bool ReconstructSecretsKernel::checkReachThreshold(const std::vector<std::string> update_model_clients,
                                                   const int cur_iterator, const std::string next_req_time,
                                                   std::shared_ptr<FBBuilder> fbb, const std::string fl_id) {
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(INFO) << "Current amount for ReconstructSecretsKernel is enough.";
    clock_t start_time = clock();
    fl::ClientNoises clients_noises_pb;
    auto ret = fl::cache::ClientInfos::GetInstance().GetClientNoises(&clients_noises_pb);
    if (ret == fl::cache::kCacheNil) {
      MS_LOG(INFO) << "Success, the secret will be reconstructed.";
      if (cipher_reconstruct_.ReconstructSecretsGenNoise(update_model_clients)) {
        cipher_reconstruct_.BuildReconstructSecretsRsp(
          fbb, schema::ResponseCode_SUCCEED, "Success,the secret is reconstructing.", cur_iterator, next_req_time);
        MS_LOG(INFO) << "CipherReconStruct::ReconstructSecrets" << fl_id << " Success, reconstruct ok.";
      } else {
        cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_OutOfTime,
                                                       "the secret restructs failed.", cur_iterator, next_req_time);
        MS_LOG(ERROR) << "CipherReconStruct::ReconstructSecrets" << fl_id << " failed.";
        return false;
      }
    } else if (ret.IsSuccess()) {
      cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_SUCCEED, "Clients' number is full.",
                                                     cur_iterator, next_req_time);
      MS_LOG(INFO) << "CipherReconStruct::ReconstructSecrets" << fl_id << " Success : no need reconstruct.";
    } else {
      cipher_reconstruct_.BuildReconstructSecretsRsp(
        fbb, schema::ResponseCode_OutOfTime, "Get client noises from cache failed.", cur_iterator, next_req_time);
      MS_LOG(ERROR) << "Get client noises from cache failed.";
      return false;
    }
    clock_t end_time = clock();
    double duration = static_cast<double>((end_time - start_time) * 1.0 / CLOCKS_PER_SEC);
    MS_LOG(INFO) << "Reconstruct get + gennoise data time is : " << duration;
  } else {
    cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_SUCCEED,
                                                   "Success, but the server is not ready to reconstruct secret yet.",
                                                   cur_iterator, std::to_string(CURRENT_TIME_MILLI.count()));
  }
  return true;
}

void ReconstructSecretsKernel::OnLastCountEvent() {
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << cache::InstanceContext::Instance().iteration_num();
  if (FLContext::instance()->encrypt_type() == kPWEncryptType) {
    Executor::GetInstance().TodoUnmask();
  }
}

REG_ROUND_KERNEL(reconstructSecrets, ReconstructSecretsKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
