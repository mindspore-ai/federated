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

#ifndef MINDSPORE_FEDERATED_PSI_H
#define MINDSPORE_FEDERATED_PSI_H

#include <vector>
#include <memory>
#include <string>

#include "armour/base_crypto/ecc.h"
#include "armour/base_crypto/bloom_filter.h"

#include "common/utils/visible.h"

namespace mindspore {
namespace fl {
namespace psi {

struct PsiCtx {
  bool SetRole(size_t peer_dataset_size) {
    if (peer_dataset_size == 0) {
      MS_LOG(WARNING) << "Context peer_dataset_size is 0, please check!";
    }
    peer_num = peer_dataset_size;
    if (self_num == 0) {
      MS_LOG(ERROR) << "PSI_Ctx is not set.";
      return false;
    }
    if (self_num <= peer_dataset_size) {
      role = "bob";
      peer_role = "alice";
    } else {
      role = "alice";
      peer_role = "bob";
    }
    return true;
  }

  bool SetRole(std::string peer_role_str) {
    if (peer_role_str == "alice") {
      role = "bob";
      peer_role = "alice";
    } else if (peer_role_str == "bob") {
      role = "alice";
      peer_role = "bob";
    } else {
      MS_LOG(ERROR) << "PSI_Ctx is not set.";
      return false;
    }
    return true;
  }

  bool CheckPsiCtxOK() {
    bool ret = true;
    if (compress_length != LENGTH_32 && compress_length != LENGTH_33) {
      ret = false;
      MS_LOG(INFO) << "Compress_length can only be set to " << LENGTH_32 << " or " << LENGTH_33;
    }

    if (compare_length < LENGTH_12 || compare_length > LENGTH_32) {
      ret = false;
      MS_LOG(INFO) << "Compare_length should be in [12, 32], but get " << compare_length;
    }

    if (psi_type == "Filter_ecdh" && compare_length != LENGTH_32) {
      ret = false;
      MS_LOG(INFO) << "If use filter ecdh, compare length must be 32, but get " << compare_length;
    }
    return ret;
  }

  std::shared_ptr<ECC> ecc;
  size_t compress_length = LENGTH_32;  // default
  size_t compare_length = LENGTH_12;   // default
  size_t thread_num = 1;
  size_t bin_id = 1;
  std::string curve_name = "p256";       // default
  std::string psi_type = "filter_ecdh";  // default
  std::string role = "alice";
  std::string peer_role = "bob";
  int neg_log_fp_rate = 40;  // default
  size_t chunk_size = 1;     // default
  bool need_check = false;

  std::string file_path = "";
  std::vector<std::string> input_vct;
  std::vector<std::string> input_hash_vct;
  size_t self_num = 0;
  size_t peer_num = 0;
};

void FindWrong(const PsiCtx &psi_ctx, const std::vector<std::string> &align_result, std::vector<std::string> *wrong_vct,
               std::vector<std::string> *fix_vct);

void DelWrong(std::vector<std::string> *align_results_vector, const std::vector<std::string> &recv_wrong_vct);

MS_EXPORT std::vector<std::string> RunPsiDemo(const std::vector<std::string> &alice_input,
                                              const std::vector<std::string> &bob_input);

void RunEcdhPsi(const PsiCtx &psi_ctx_alice, const PsiCtx &psi_ctx_bob);

void RunInverseEcdhPsi(const PsiCtx &psi_ctx_alice, const PsiCtx &psi_ctx_bob);

std::vector<std::string> RunInverseFilterEcdhPsi(const PsiCtx &psi_ctx_alice, const PsiCtx &psi_ctx_bob);

std::vector<std::string> CreateRangeItems(size_t begin, size_t size);

MS_EXPORT std::vector<std::string> RunPSI(const std::vector<std::string> &input_vct, const size_t bin_id,
                                          const std::string &COM_role, const std::string &ip,
                                          const std::string &psi_type, size_t thread_num, bool need_check);

}  // namespace psi
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_FEDERATED_PSI_H
