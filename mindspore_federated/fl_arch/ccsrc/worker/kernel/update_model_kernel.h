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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_UPDATE_MODEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_UPDATE_MODEL_H_

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <map>

#include "worker/worker.h"
#include "armour/secure_protocol/masking.h"
#include "common/core/comm_util.h"
#include "common/exit_handler.h"

namespace mindspore {
namespace fl {
namespace worker {
namespace kernel {
class UpdateModelKernelMod : public AbstractKernel {
 public:
  UpdateModelKernelMod() = default;
  ~UpdateModelKernelMod() override = default;

  static std::shared_ptr<UpdateModelKernelMod> GetInstance() {
    static std::shared_ptr<UpdateModelKernelMod> instance = nullptr;
    if (instance == nullptr) {
      instance.reset(new UpdateModelKernelMod());
      instance->Init();
    }
    return instance;
  }

  bool Launch(std::map<std::string, std::vector<float>> *weight_datas) {
    MS_LOG(INFO) << "Launching client UpdateModelKernelMod";
    MS_EXCEPTION_IF_NULL(weight_datas);

    if (!WeightingData(weight_datas)) {
      MS_LOG(EXCEPTION) << "Weighting data with data_size failed.";
      return false;
    }

    if (encrypt_type_.compare("STABLE_PW_ENCRYPT") == 0) {
      EncryptData(weight_datas);
    }
    FBBuilder builder;
    if (!BuildUpdateModelReq(&builder, *weight_datas)) {
      MS_LOG(EXCEPTION) << "Building request for FusedPushWeight failed.";
      return false;
    }
    std::shared_ptr<std::vector<unsigned char>> update_model_rsp_msg = nullptr;
    if (!fl::worker::Worker::GetInstance().SendToServer(builder.GetBufferPointer(), builder.GetSize(),
                                                        fl::TcpUserCommand::kUpdateModel, &update_model_rsp_msg)) {
      MS_LOG(EXCEPTION) << "Sending request for UpdateModel to server failed.";
      return false;
    }
    flatbuffers::Verifier verifier(update_model_rsp_msg->data(), update_model_rsp_msg->size());
    if (!verifier.VerifyBuffer<schema::ResponseUpdateModel>()) {
      MS_LOG(EXCEPTION) << "The schema of ResponseUpdateModel is invalid.";
      return false;
    }

    const schema::ResponseFLJob *update_model_rsp =
      flatbuffers::GetRoot<schema::ResponseFLJob>(update_model_rsp_msg->data());
    MS_EXCEPTION_IF_NULL(update_model_rsp);
    auto response_code = update_model_rsp->retcode();
    switch (response_code) {
      case schema::ResponseCode_SUCCEED:
      case schema::ResponseCode_OutOfTime:
        break;
      default:
        MS_LOG(EXCEPTION) << "Launching start fl job for worker failed. Reason: " << update_model_rsp->reason();
    }
    return true;
  }

  void Init() override {
    MS_LOG(INFO) << "Initializing UpdateModel kernel";
    fl_name_ = fl::worker::Worker::GetInstance().fl_name();
    fl_id_ = fl::worker::Worker::GetInstance().fl_id();
    data_size_ = fl::worker::Worker::GetInstance().data_size();
    encrypt_type_ = fl::worker::Worker::GetInstance().encrypt_type();
    if (encrypt_type_.compare("NOT_ENCRYPT") != 0 && encrypt_type_.compare("STABLE_PW_ENCRYPT") != 0) {
      MS_LOG(EXCEPTION)
        << "Value Error: the parameter 'encrypt_type' of updateModel kernel can only be 'NOT_ENCRYPT' or "
           "'STABLE_PW_ENCRYPT' until now, but got: "
        << encrypt_type_;
    }

    if (encrypt_type_.compare("STABLE_PW_ENCRYPT") == 0) {
      MS_LOG(INFO) << "STABLE_PW_ENCRYPT mode is open, model weights will be encrypted before send to server.";
      client_keys = fl::worker::Worker::GetInstance().public_keys_list();
      if (client_keys.size() == 0) {
        MS_LOG(EXCEPTION) << "The size of local-stored client_keys_list is 0, please check whether P.ExchangeKeys() "
                             "and P.GetKeys() have been executed before updateModel.";
      }
    }
    MS_LOG(INFO) << "Initializing StartFLJob kernel. fl_name: " << fl_name_ << ", fl_id: " << fl_id_
                 << ". Request will be sent to server, Encrypt type: " << encrypt_type_;
  }

 private:
  bool BuildUpdateModelReq(FBBuilder *fbb, const std::map<std::string, std::vector<float>> &weight_datas) {
    MS_EXCEPTION_IF_NULL(fbb);
    auto fbs_fl_name = fbb->CreateString(fl_name_);
    auto fbs_fl_id = fbb->CreateString(fl_id_);
    auto time = fl::CommUtil::GetNowTime();
    MS_LOG(INFO) << "now time: " << time.time_str_mill;
    auto fbs_timestamp = fbb->CreateString(std::to_string(time.time_stamp));
    std::vector<flatbuffers::Offset<schema::FeatureMap>> fbs_feature_maps;
    for (auto &weight_item : weight_datas) {
      const std::string &weight_name = weight_item.first;
      auto &weight_data = weight_item.second;
      auto fbs_weight_fullname = fbb->CreateString(weight_name);
      auto fbs_weight_data = fbb->CreateVector(weight_data);
      auto fbs_feature_map = schema::CreateFeatureMap(*fbb, fbs_weight_fullname, fbs_weight_data);
      fbs_feature_maps.push_back(fbs_feature_map);
    }
    auto fbs_feature_maps_vector = fbb->CreateVector(fbs_feature_maps);

    schema::RequestUpdateModelBuilder req_update_model_builder(*fbb);
    req_update_model_builder.add_fl_name(fbs_fl_name);
    req_update_model_builder.add_fl_id(fbs_fl_id);
    req_update_model_builder.add_timestamp(fbs_timestamp);
    iteration_ = fl::worker::Worker::GetInstance().fl_iteration_num();
    req_update_model_builder.add_iteration(SizeToInt(iteration_));
    req_update_model_builder.add_feature_map(fbs_feature_maps_vector);
    auto req_update_model = req_update_model_builder.Finish();
    fbb->Finish(req_update_model);
    return true;
  }

  bool WeightingData(std::map<std::string, std::vector<float>> *weight_datas) {
    for (auto &item : *weight_datas) {
      auto &data = item.second;
      for (size_t i = 0; i < data.size(); i++) {
        data[i] *= data_size_;
      }
    }
    return true;
  }

  void EncryptData(std::map<std::string, std::vector<float>> *weight_datas) {
    // calculate the sum of all layer's weight size
    size_t total_size = 0;
    for (auto &weight_item : *weight_datas) {
      auto &weight_data = weight_item.second;
      total_size += weight_data.size();
    }
    // get pairwise encryption noise vector
    std::vector<float> noise_vector = GetEncryptNoise(total_size);

    // encrypt original data
    size_t encrypt_num = 0;
    for (auto &weight_item : *weight_datas) {
      const std::string &weight_name = weight_item.first;
      auto &original_data = weight_item.second;
      MS_LOG(INFO) << "Encrypt weights of layer: " << weight_name;
      size_t weights_size = original_data.size();
      for (size_t j = 0; j < weights_size; j++) {
        original_data[j] += noise_vector[j + encrypt_num];
      }
      encrypt_num += weights_size;
    }
    MS_LOG(INFO) << "Encrypt data finished.";
  }

  // compute the pairwise noise based on local worker's private key and remote workers' public key
  std::vector<float> GetEncryptNoise(size_t noise_len) {
    std::vector<float> total_noise(noise_len, 0);
    int client_num = client_keys.size();
    for (int i = 0; i < client_num; i++) {
      EncryptPublicKeys public_key_set_i = client_keys[i];
      std::string remote_fl_id = public_key_set_i.flID;
      // do not need to compute pairwise noise with itself
      if (remote_fl_id == fl_id_) {
        continue;
      }
      // get local worker's private key
      armour::PrivateKey *local_private_key = fl::worker::Worker::GetInstance().secret_pk();
      if (local_private_key == nullptr) {
        MS_LOG(EXCEPTION) << "Local secret private key is nullptr, get encryption noise failed!";
      }

      // choose pw_iv and pw_salt for encryption, we choose that of smaller fl_id worker's
      std::vector<uint8_t> encrypt_pw_iv;
      std::vector<uint8_t> encrypt_pw_salt;
      if (fl_id_ < remote_fl_id) {
        encrypt_pw_iv = fl::worker::Worker::GetInstance().pw_iv();
        encrypt_pw_salt = fl::worker::Worker::GetInstance().pw_salt();
      } else {
        encrypt_pw_iv = public_key_set_i.pwIV;
        encrypt_pw_salt = public_key_set_i.pwSalt;
      }

      // get keyAgreement seed
      std::vector<uint8_t> remote_public_key = public_key_set_i.publicKey;
      armour::PublicKey *pubKey =
        armour::KeyAgreement::FromPublicBytes(remote_public_key.data(), remote_public_key.size());
      uint8_t secret1[SECRET_MAX_LEN] = {0};
      int ret = armour::KeyAgreement::ComputeSharedKey(
        local_private_key, pubKey, SECRET_MAX_LEN, encrypt_pw_salt.data(), SizeToInt(encrypt_pw_salt.size()), secret1);
      delete pubKey;
      if (ret < 0) {
        MS_LOG(EXCEPTION) << "Get secret seed failed!";
      }

      // generate pairwise encryption noise
      std::vector<float> noise_i;
      if (armour::Masking::GetMasking(&noise_i, noise_len, (const uint8_t *)secret1, SECRET_MAX_LEN,
                                      encrypt_pw_iv.data(), encrypt_pw_iv.size()) < 0) {
        MS_LOG(EXCEPTION) << "Get masking noise failed.";
      }
      int noise_sign = (fl_id_ < remote_fl_id) ? -1 : 1;
      for (size_t k = 0; k < noise_len; k++) {
        total_noise[k] += noise_sign * noise_i[k];
      }
      MS_LOG(INFO) << "Generate noise between fl_id: " << fl_id_ << " and fl_id: " << remote_fl_id << " finished.";
    }
    return total_noise;
  }
  std::string fl_name_;
  std::string fl_id_;
  int data_size_;
  uint64_t iteration_;
  std::vector<EncryptPublicKeys> client_keys;
  std::string encrypt_type_;
};
}  // namespace kernel
}  // namespace worker
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_UPDATE_MODEL_H_
