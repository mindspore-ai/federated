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

#include "worker/kernel/get_keys_kernel.h"

namespace mindspore {
namespace fl {
namespace worker {
namespace kernel {
bool GetKeysKernelMod::Launch() {
  MS_LOG(INFO) << "Launching client GetKeysKernelMod";
  FBBuilder fbb;
  BuildGetKeysReq(&fbb);

  if (!fl::worker::CloudWorker::GetInstance().SendToServerSync(kernel_path_, HTTP_CONTENT_TYPE_URL_ENCODED,
                                                               fbb.GetBufferPointer(), fbb.GetSize())) {
    MS_LOG(WARNING) << "Sending request for getKeys to server failed.";
    return false;
  }

  MS_LOG(INFO) << "Get keys successfully.";
  return true;
}

void GetKeysKernelMod::Init() {
  fl_id_ = fl::worker::CloudWorker::GetInstance().fl_id();

  kernel_path_ = "/getKeys";
  MS_LOG(INFO) << "Initializing GetKeys kernel"
               << ", fl_id: " << fl_id_;

  fl::worker::CloudWorker::GetInstance().RegisterMessageCallback(
    kernel_path_, [&](const std::shared_ptr<std::vector<unsigned char>> &response_msg) {
      if (response_msg == nullptr) {
        MS_LOG(EXCEPTION) << "Received message pointer is nullptr.";
        return;
      }
      flatbuffers::Verifier verifier(response_msg->data(), response_msg->size());
      if (!verifier.VerifyBuffer<schema::ReturnExchangeKeys>()) {
        MS_LOG(WARNING) << "The schema of response message is invalid.";
        return;
      }
      const schema::ReturnExchangeKeys *get_keys_rsp =
        flatbuffers::GetRoot<schema::ReturnExchangeKeys>(response_msg->data());
      MS_EXCEPTION_IF_NULL(get_keys_rsp);
      auto response_code = get_keys_rsp->retcode();
      if ((response_code != schema::ResponseCode_SUCCEED) && (response_code != schema::ResponseCode_OutOfTime)) {
        MS_LOG(EXCEPTION) << "Launching get keys job for worker failed. response_code: " << response_code;
      }

      bool save_keys_succeed = SavePublicKeyList(get_keys_rsp->remote_publickeys());
      if (!save_keys_succeed) {
        MS_LOG(EXCEPTION) << "Save received remote keys failed.";
      }
      return;
    });

  MS_LOG(INFO) << "Initialize GetKeys kernel successfully.";
}

void GetKeysKernelMod::BuildGetKeysReq(FBBuilder *fbb) {
  MS_EXCEPTION_IF_NULL(fbb);
  int iter = fl::worker::CloudWorker::GetInstance().fl_iteration_num();
  auto fbs_fl_id = fbb->CreateString(fl_id_);
  schema::GetExchangeKeysBuilder get_keys_builder(*fbb);
  get_keys_builder.add_fl_id(fbs_fl_id);
  get_keys_builder.add_iteration(iter);
  auto req_fl_job = get_keys_builder.Finish();
  fbb->Finish(req_fl_job);
  MS_LOG(INFO) << "BuildGetKeysReq successfully.";
}

bool GetKeysKernelMod::SavePublicKeyList(
  const flatbuffers::Vector<flatbuffers::Offset<schema::ClientPublicKeys>> *remote_public_key) {
  if (remote_public_key == nullptr) {
    MS_LOG(EXCEPTION) << "Input remote_pubic_key is nullptr.";
  }

  int client_num = remote_public_key->size();
  if (client_num <= 0) {
    MS_LOG(EXCEPTION) << "Received client keys length is <= 0, please check it!";
    return false;
  }

  // save client keys list
  std::vector<EncryptPublicKeys> saved_remote_public_keys;
  for (auto iter = remote_public_key->begin(); iter != remote_public_key->end(); ++iter) {
    std::string fl_id = iter->fl_id()->str();
    auto fbs_spk = iter->s_pk();
    auto fbs_pw_iv = iter->pw_iv();
    auto fbs_pw_salt = iter->pw_salt();
    if (fbs_spk == nullptr || fbs_pw_iv == nullptr || fbs_pw_salt == nullptr) {
      MS_LOG(WARNING) << "public key, pw_iv or pw_salt in remote_publickeys is nullptr.";
    } else {
      std::vector<uint8_t> spk_vector;
      std::vector<uint8_t> pw_iv_vector;
      std::vector<uint8_t> pw_salt_vector;
      spk_vector.assign(fbs_spk->begin(), fbs_spk->end());
      pw_iv_vector.assign(fbs_pw_iv->begin(), fbs_pw_iv->end());
      pw_salt_vector.assign(fbs_pw_salt->begin(), fbs_pw_salt->end());
      EncryptPublicKeys public_keys_i;
      public_keys_i.flID = fl_id;
      public_keys_i.publicKey = spk_vector;
      public_keys_i.pwIV = pw_iv_vector;
      public_keys_i.pwSalt = pw_salt_vector;
      saved_remote_public_keys.push_back(public_keys_i);
      MS_LOG(INFO) << "Add public keys of client:" << fl_id << " successfully.";
    }
  }
  fl::worker::CloudWorker::GetInstance().set_public_keys_list(saved_remote_public_keys);
  return true;
}
}  // namespace kernel
}  // namespace worker
}  // namespace fl
}  // namespace mindspore
