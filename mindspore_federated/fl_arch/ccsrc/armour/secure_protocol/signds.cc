/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "armour/secure_protocol/signds.h"

#include <string>
#include <memory>
#include <vector>
#include <cmath>
#include "common/constants.h"
#include "common/common.h"
#include "distributed_cache/redis_keys.h"

namespace mindspore {
namespace fl {
namespace cache {

// Restoring the True Frequency of Disturbances in Random Response Algorithms.
size_t SignDS::ComputRandomResponseB(const std::vector<std::string> &all_b_hat) {
  size_t count_0 = 0;
  size_t count_1 = 0;
  for (std::string b_hat : all_b_hat) {
    if (b_hat == kSignDSbHat0) {
      count_0 += 1;
    } else if (b_hat == kSignDSbHat1) {
      count_1 += 1;
    }
  }
  float probability_keep = exp(kSignRREps) / (1 + exp(kSignRREps));
  float random_response_number = 2.0;
  float real_count_1 =
    (count_1 + (count_1 + count_0) * (probability_keep - 1)) / (random_response_number * probability_keep - 1);
  float real_count_0 = count_0 + count_1 - real_count_1;
  MS_LOG_INFO << "real_count_1 num is " << real_count_1 << " real_count_0 num is " << real_count_0;
  size_t mag_b = real_count_1 > real_count_0 ? 1 : 0;
  return mag_b;
}

// If the old client accounts for the majority, then the global learning rate parameter is still used when
// reconstructing the gradient.
bool SignDS::CheckOldVersion(const std::vector<std::string> &all_b_hat, uint64_t is_reached, size_t updatemodel_num) {
  if (updatemodel_num == 0) {
    MS_LOG_WARNING << "updatemodel num is 0, please check!";
    return false;
  }
  MS_LOG_INFO << "The number of bHat is " << all_b_hat.size() << ", the number of updatedemodel is " << updatemodel_num;
  auto enc_config = FLContext::instance()->encrypt_config();
  auto client = cache::DistributedCacheLoader::Instance().GetOneClient();
  auto redis_key_instance = cache::RedisKeys::GetInstance();
  if (all_b_hat.size() < kSignOldRatioUpper * updatemodel_num) {
    MS_LOG_WARNING << "The number of bHat is too small, use global lr: " << enc_config.sign_global_lr;
    if (is_reached == 0) {
      client->Set(redis_key_instance.ClientSignDSrEstHash(),
                  std::to_string(enc_config.sign_global_lr / kSignDSGlobalLRRatioOfNotReached / updatemodel_num));
    } else if (is_reached == 1) {
      client->Set(redis_key_instance.ClientSignDSrEstHash(),
                  std::to_string(enc_config.sign_global_lr / kSignDSGlobalLRRatioOfReached / updatemodel_num));
    }
    return true;
  }
  return false;
}


// Update the global parameters of SignDS.
void SignDS::SummarizeSignDS() {
  if (FLContext::instance() != nullptr && FLContext::instance()->encrypt_config().encrypt_type != kDSEncryptType) {
    return;
  }
  std::vector<std::string> all_b_hat;
  GetSignDSbHatAndReset(&all_b_hat);

  uint64_t is_reached;
  auto client = cache::DistributedCacheLoader::Instance().GetOneClient();
  auto redis_key_instance = cache::RedisKeys::GetInstance();
  client->Get(redis_key_instance.ClientSignDSIsReachedHash(), kSignInitIsNotReached, &is_reached);
  MS_LOG_INFO << "Before change, is_reached is " << is_reached;
  size_t updatemodel_num =
    FLContext::instance()->update_model_ratio() * FLContext::instance()->start_fl_job_threshold();
  if (CheckOldVersion(all_b_hat, is_reached, updatemodel_num)) {
    return;
  }
  size_t mag_b = ComputRandomResponseB(all_b_hat);
  uint64_t count_step = 1;
  uint64_t reached = 1;
  float r_est;
  uint64_t reach_count;
  client->GetFloat(redis_key_instance.ClientSignDSrEstHash(), kSignInitREst, &r_est);
  MS_LOG_INFO << "Before change, rEst is " << r_est;

  if (is_reached == 0 && mag_b == 0) {
    r_est *= kSignExpansionFactor;
    MS_LOG_INFO << "After expansion, rEst is " << r_est;
  } else if (is_reached == 0 && mag_b == 1) {
    client->Get(redis_key_instance.ClientSignDSReachedCountHash(), kSignInitReachedCount, &reach_count);
    client->Incr(redis_key_instance.ClientSignDSReachedCountHash(), &count_step);
    MS_LOG_INFO << "reach count is " << reach_count;
    if (reach_count >= kSignReachedThreshold) {
      client->Incr(redis_key_instance.ClientSignDSIsReachedHash(), &reached);
      client->Get(redis_key_instance.ClientSignDSIsReachedHash(), kSignInitIsNotReached, &is_reached);
      MS_LOG_INFO << "sign_is_reached is " << is_reached;
    }
  } else if (is_reached == 1 && mag_b == 1) {
    if (kSignReductionFactor <= 0) {
      MS_LOG_WARNING << "sign_reduction_factor set ERROR, please check";
      return;
    }
    r_est /= kSignReductionFactor;
    MS_LOG_INFO << "After reduction, rEst is " << r_est;
  }
  client->Set(redis_key_instance.ClientSignDSrEstHash(), std::to_string(r_est));
  float signds_grad = is_reached == 0 ? kSignDSGlobalLRRatioOfNotReached * updatemodel_num * r_est
                                      : kSignDSGlobalLRRatioOfReached * updatemodel_num * r_est;
  MS_LOG_INFO << "signds_grad is " << signds_grad;
}

// Obtain feedback from all devices, save the result to vector, and clear the Redis.
void SignDS::GetSignDSbHatAndReset(std::vector<std::string> *all_b_hat) {
  if (all_b_hat == nullptr) {
    MS_LOG_WARNING << "Dst_vector is null, please check.";
    return;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed.";
    return;
  }
  auto name = cache::RedisKeys::GetInstance().ClientSignDSbHatHash();
  std::vector<std::string> items;
  size_t length;
  auto status_llen = client->LLen(name, &length);
  if (!status_llen.IsSuccess()) {
    MS_LOG_WARNING << "Redis LLen failed.";
    return;
  }
  if (length == 0) {
    return;
  }
  auto status = client->LRange(name, 0, length - 1, &items);
  if (status.IsSuccess() && !items.empty()) {
    for (auto &item : items) {
      (*all_b_hat).push_back(item);
    }
  }
  auto status_ltrim = client->LTrim(name, 1, 0);
  if (!status_ltrim.IsSuccess()) {
    MS_LOG_WARNING << "Ltrim failed";
    return;
  }
}

float SignDS::GetREst() {
  float sign_r_est;
  auto client = cache::DistributedCacheLoader::Instance().GetOneClient();
  auto redis_key_instance = cache::RedisKeys::GetInstance();
  auto signds = cache::SignDS::Instance();
  client->GetFloat(redis_key_instance.ClientSignDSrEstHash(), signds.kSignInitREst, &sign_r_est);
  return sign_r_est;
}

uint64_t SignDS::GetIsReached() {
  uint64_t sign_is_reached;
  auto client = cache::DistributedCacheLoader::Instance().GetOneClient();
  auto redis_key_instance = cache::RedisKeys::GetInstance();
  auto signds = cache::SignDS::Instance();
  client->Get(redis_key_instance.ClientSignDSIsReachedHash(), signds.kSignInitIsNotReached, &sign_is_reached);
  return sign_is_reached;
}

}  // namespace cache
}  // namespace fl
}  // namespace mindspore
