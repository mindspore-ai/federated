/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * secGear is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *     http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
 * PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <linux/limits.h>
#include <openssl/ec.h>
#include <utility>
#include <vector>

#ifdef ENABLE_TEE
#include <secGear/status.h>
#include "armour/secure_protocol/enclave_call.h"
#endif

#include "common/utils/visible.h"
#define BUF_LEN 32

namespace mindspore {
namespace fl {
namespace TEE {

MS_EXPORT int init_tee_cut_layer(size_t batch_size, size_t featureA_dims, size_t featureB_dims, size_t output_dims,
                                 float learning_rate, float loss_scale);
MS_EXPORT std::vector<float> forward_tee_cut_layer(size_t batch_size, size_t featureA_dims, size_t featureB_dims,
                                                   std::vector<float> *embA, std::vector<float> *embB,
                                                   size_t output_dims);
MS_EXPORT std::vector<std::vector<float>> backward_tee_cut_layer(size_t batch_size, size_t featureA_dims,
                                                                 size_t featurB_dims, size_t output_dims,
                                                                 std::vector<float> *d_output);
MS_EXPORT std::pair<std::vector<uint8_t>, int> encrypt_client_data(std::vector<float> *plain, size_t plain_len);
MS_EXPORT std::vector<float> secure_forward_tee_cut_layer(size_t batch_size, size_t featureA_dims,
                                                          size_t featureB_dims, std::vector<uint8_t> *encrypted_embA,
                                                          size_t encrypted_embA_len,
                                                          std::vector<uint8_t> *encrypted_embB,
                                                          size_t encrypted_embB_len, size_t output_dims);
MS_EXPORT int free_tee_cut_layer();

}  // namespace TEE
}  // namespace fl
}  // namespace mindspore
