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
#include "distributed_cache/hyper_params.h"
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "distributed_cache/redis_keys.h"
#include "distributed_cache/distributed_cache.h"
#include "distributed_cache/timer.h"
#include "distributed_cache/scheduler.h"
#include "common/fl_context.h"

namespace mindspore {
namespace fl {
namespace cache {
namespace {
#define DEFINE_HYPER_VAR(name) const char *kHyper_##name = #name;
#define HYPER_VAR(name) kHyper_##name

DEFINE_HYPER_VAR(global_iteration_time_window)
DEFINE_HYPER_VAR(start_fl_job_threshold)
DEFINE_HYPER_VAR(start_fl_job_time_window)
DEFINE_HYPER_VAR(update_model_ratio)
DEFINE_HYPER_VAR(update_model_time_window)
DEFINE_HYPER_VAR(client_epoch_num)
DEFINE_HYPER_VAR(client_batch_size)
DEFINE_HYPER_VAR(client_learning_rate)
DEFINE_HYPER_VAR(fl_iteration_num)

// cipher, for round
DEFINE_HYPER_VAR(encrypt_type)
DEFINE_HYPER_VAR(share_secrets_ratio)
DEFINE_HYPER_VAR(cipher_time_window)
DEFINE_HYPER_VAR(reconstruct_secrets_threshold)
// cipher
DEFINE_HYPER_VAR(secure_aggregation)
DEFINE_HYPER_VAR(dp_eps)
DEFINE_HYPER_VAR(dp_delta)
DEFINE_HYPER_VAR(dp_norm_clip)
DEFINE_HYPER_VAR(sign_k)
DEFINE_HYPER_VAR(sign_eps)
DEFINE_HYPER_VAR(sign_thr_ratio)
DEFINE_HYPER_VAR(sign_global_lr)
DEFINE_HYPER_VAR(sign_dim_out)

// compress
DEFINE_HYPER_VAR(upload_compress_type)
DEFINE_HYPER_VAR(upload_sparse_rate)
DEFINE_HYPER_VAR(download_compress_type)

DEFINE_HYPER_VAR(enable_ssl)
DEFINE_HYPER_VAR(pki_verify)
}  // namespace

CacheStatus HyperParams::InitAndSync() { return SyncOnNewInstance(); }

// on init and new instance
CacheStatus HyperParams::SyncOnNewInstance() {
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  auto status = SyncLocal2Cache(client);
  if (status == kCacheExist) {
    status = SyncCache2Local(client);
  }
  if (status.IsSuccess()) {
    auto context = FLContext::instance();
    if (context == nullptr) {
      return kCacheInnerErr;
    }
    MS_LOG(INFO) << "start_fl_job_threshold: " << context->start_fl_job_threshold();
    MS_LOG(INFO) << "start_fl_job_time_window in ms: " << context->start_fl_job_time_window();
    MS_LOG(INFO) << "update_model_ratio: " << context->update_model_ratio();
    MS_LOG(INFO) << "update_model_time_window in ms: " << context->update_model_time_window();
    MS_LOG(INFO) << "fl_iteration_num: " << context->fl_iteration_num();
    MS_LOG(INFO) << "client_epoch_num: " << context->client_epoch_num();
    MS_LOG(INFO) << "client_batch_size: " << context->client_batch_size();
    MS_LOG(INFO) << "client_learning_rate: " << context->client_learning_rate();
    MS_LOG(INFO) << "global_iteration_time_window in ms: " << context->global_iteration_time_window();
  }
  return status;
}

CacheStatus HyperParams::SyncPeriod() {
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  return SyncLocal2Cache(client);
}

CacheStatus HyperParams::SyncLocal2Cache(const std::shared_ptr<RedisClientBase> &client) {
  auto context = FLContext::instance();
  if (context == nullptr) {
    return kCacheInnerErr;
  }
  if (client == nullptr) {
    return kCacheNetErr;
  }
  nlohmann::json obj;
  obj[HYPER_VAR(global_iteration_time_window)] = context->global_iteration_time_window();
  obj[HYPER_VAR(start_fl_job_threshold)] = context->start_fl_job_threshold();
  obj[HYPER_VAR(start_fl_job_time_window)] = context->start_fl_job_time_window();
  obj[HYPER_VAR(update_model_ratio)] = context->update_model_ratio();
  obj[HYPER_VAR(update_model_time_window)] = context->update_model_time_window();
  obj[HYPER_VAR(client_epoch_num)] = context->client_epoch_num();
  obj[HYPER_VAR(client_batch_size)] = context->client_batch_size();
  obj[HYPER_VAR(client_learning_rate)] = context->client_learning_rate();
  obj[HYPER_VAR(fl_iteration_num)] = context->fl_iteration_num();

  // cipher
  obj[HYPER_VAR(secure_aggregation)] = context->secure_aggregation();

  // cipher, for round
  auto &encrypt_config = context->encrypt_config();
  obj[HYPER_VAR(encrypt_type)] = encrypt_config.encrypt_type;
  obj[HYPER_VAR(share_secrets_ratio)] = encrypt_config.share_secrets_ratio;
  obj[HYPER_VAR(cipher_time_window)] = encrypt_config.cipher_time_window;
  obj[HYPER_VAR(reconstruct_secrets_threshold)] = encrypt_config.reconstruct_secrets_threshold;

  obj[HYPER_VAR(dp_eps)] = encrypt_config.dp_eps;
  obj[HYPER_VAR(dp_delta)] = encrypt_config.dp_delta;
  obj[HYPER_VAR(dp_norm_clip)] = encrypt_config.dp_norm_clip;
  obj[HYPER_VAR(sign_k)] = encrypt_config.sign_k;
  obj[HYPER_VAR(sign_eps)] = encrypt_config.sign_eps;
  obj[HYPER_VAR(sign_thr_ratio)] = encrypt_config.sign_thr_ratio;
  obj[HYPER_VAR(sign_global_lr)] = encrypt_config.sign_global_lr;
  obj[HYPER_VAR(sign_dim_out)] = encrypt_config.sign_dim_out;

  // compress
  auto &compression_config = context->compression_config();
  obj[HYPER_VAR(upload_compress_type)] = compression_config.upload_compress_type;
  obj[HYPER_VAR(upload_sparse_rate)] = compression_config.upload_sparse_rate;
  obj[HYPER_VAR(download_compress_type)] = compression_config.download_compress_type;

  obj[HYPER_VAR(enable_ssl)] = context->enable_ssl();
  obj[HYPER_VAR(pki_verify)] = context->pki_verify();

  auto val = obj.dump();
  auto key = RedisKeys::GetInstance().HyperParamsString();
  auto result = client->SetExNx(key, val, Timer::config_expire_time_in_seconds());
  if (result.IsSuccess()) {
    MS_LOG_INFO << "Sync hyper params to cache success";
  }
  return result;
}

CacheStatus HyperParams::SyncCache2Local(const std::shared_ptr<RedisClientBase> &client) {
  auto context = FLContext::instance();
  if (context == nullptr) {
    return kCacheInnerErr;
  }
  if (client == nullptr) {
    return kCacheNetErr;
  }
  auto key = RedisKeys::GetInstance().HyperParamsString();
  std::string val;
  auto status = client->Get(key, &val);
  if (!status.IsSuccess()) {
    return status;
  }
  nlohmann::json obj;
  try {
    obj = nlohmann::json::parse(val);

    context->set_global_iteration_time_window(obj[HYPER_VAR(global_iteration_time_window)]);
    context->set_start_fl_job_threshold(obj[HYPER_VAR(start_fl_job_threshold)]);
    context->set_start_fl_job_time_window(obj[HYPER_VAR(start_fl_job_time_window)]);
    context->set_update_model_ratio(obj[HYPER_VAR(update_model_ratio)]);
    context->set_update_model_time_window(obj[HYPER_VAR(update_model_time_window)]);
    context->set_client_epoch_num(obj[HYPER_VAR(client_epoch_num)]);
    context->set_client_batch_size(obj[HYPER_VAR(client_batch_size)]);
    context->set_client_learning_rate(obj[HYPER_VAR(client_learning_rate)]);
    context->set_fl_iteration_num(obj[HYPER_VAR(fl_iteration_num)]);

    // cipher, for round
    EncryptConfig encrypt_config;
    encrypt_config.encrypt_type = obj[HYPER_VAR(encrypt_type)];
    encrypt_config.share_secrets_ratio = obj[HYPER_VAR(share_secrets_ratio)];
    encrypt_config.cipher_time_window = obj[HYPER_VAR(cipher_time_window)];
    encrypt_config.reconstruct_secrets_threshold = obj[HYPER_VAR(reconstruct_secrets_threshold)];

    encrypt_config.dp_eps = obj[HYPER_VAR(dp_eps)];
    encrypt_config.dp_delta = obj[HYPER_VAR(dp_delta)];
    encrypt_config.dp_delta = obj[HYPER_VAR(dp_delta)];
    encrypt_config.dp_norm_clip = obj[HYPER_VAR(dp_norm_clip)];
    encrypt_config.sign_k = obj[HYPER_VAR(sign_k)];
    encrypt_config.sign_eps = obj[HYPER_VAR(sign_eps)];
    encrypt_config.sign_thr_ratio = obj[HYPER_VAR(sign_thr_ratio)];
    encrypt_config.sign_global_lr = obj[HYPER_VAR(sign_global_lr)];
    encrypt_config.sign_dim_out = obj[HYPER_VAR(sign_dim_out)];

    context->set_encrypt_config(encrypt_config);
    context->set_secure_aggregation(obj[HYPER_VAR(secure_aggregation)]);

    // compress
    CompressionConfig compression_config;
    compression_config.upload_compress_type = obj[HYPER_VAR(upload_compress_type)];
    compression_config.upload_sparse_rate = obj[HYPER_VAR(upload_sparse_rate)];
    compression_config.download_compress_type = obj[HYPER_VAR(download_compress_type)];
    context->set_compression_config(compression_config);

    auto bool_as_str = [](bool val) -> std::string { return val ? "true" : "false"; };
    bool cache_enable_ssl = obj[HYPER_VAR(enable_ssl)];
    if (cache_enable_ssl != context->enable_ssl()) {
      MS_LOG_ERROR << "Context 'enable_ssl' " << bool_as_str(context->enable_ssl()) << " of local != that "
                   << bool_as_str(cache_enable_ssl) << " declared in distributed cache.";
      return kCacheParamFailed;
    }
    bool cache_pki_verify = obj[HYPER_VAR(pki_verify)];
    if (cache_pki_verify != context->pki_verify()) {
      MS_LOG_ERROR << "Context 'pki_verify' " << bool_as_str(context->pki_verify()) << " of local != that "
                   << bool_as_str(cache_pki_verify) << " declared in distributed cache.";
      return kCacheParamFailed;
    }
    MS_LOG_INFO << "Sync hyper params from cache success";
  } catch (const std::exception &e) {
    MS_LOG_ERROR << "Catch exception when parse hyper params: " << e.what();
    return kCacheInnerErr;
  }
  return kCacheSuccess;
}

bool HyperParams::MergeHyperJsonConfig(const std::string &fl_name, const std::string &hyper_params,
                                       std::string *error_msg, std::string *output_hyper_params) {
  if (error_msg == nullptr || output_hyper_params == nullptr) {
    return false;
  }
  std::string instance_name;
  (void)cache::Scheduler::Instance().GetInstanceName(fl_name, &instance_name);
  if (instance_name.empty()) {
    *error_msg = "Cannot find cluster info for " + fl_name;
    return false;
  }
  auto key = RedisKeys::GetInstance().HyperParamsString(fl_name, instance_name);
  std::string val;
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    *error_msg = "Failed to access the cache server. Please retry later.";
    return false;
  }
  auto status = client->Get(key, &val);
  if (!status.IsSuccess()) {
    *error_msg = "Failed to get current hyper params config";
    return false;
  }
  nlohmann::json obj_new;
  try {
    obj_new = nlohmann::json::parse(val);
  } catch (const std::exception &e) {
    *error_msg = "Failed to parse current hyper params config";
    return false;
  }
  nlohmann::json obj;
  auto check_unsigned = [&obj, &obj_new, error_msg](const std::string &filed) {
    auto it = obj.find(filed);
    if (it != obj.end()) {
      if (!it->is_number_unsigned()) {
        *error_msg = "Field " + filed + " in hyper param config should be unsigned integer number";
        return false;
      }
      obj_new[filed] = it.value();
    }
    return true;
  };
  auto check_float = [&obj, &obj_new, error_msg](const std::string &filed) {
    auto it = obj.find(filed);
    if (it != obj.end()) {
      if (!it->is_number()) {
        *error_msg = "Field " + filed + " in hyper param config should be float or integer number";
        return false;
      }
      if (it.value() <= 0) {
        *error_msg = "Field " + filed + " in hyper param config should be larger than 0";
        return false;
      }
      obj_new[filed] = it.value();
    }
    return true;
  };
  try {
    obj = nlohmann::json::parse(hyper_params);
  } catch (const std::exception &e) {
    *error_msg = "Expect hyper param config to be json object";
    return false;
  }
  if (!obj.is_object()) {
    *error_msg = "Expect hyper param config to be json object";
    return false;
  }
#define CHECK_HYPER_UINT64(name)          \
  if (!check_unsigned(HYPER_VAR(name))) { \
    return false;                         \
  }

#define CHECK_HYPER_FLOAT(name)        \
  if (!check_float(HYPER_VAR(name))) { \
    return false;                      \
  }
  // all integer value should larger than 0
  CHECK_HYPER_UINT64(global_iteration_time_window)
  CHECK_HYPER_UINT64(start_fl_job_threshold)
  CHECK_HYPER_UINT64(start_fl_job_time_window)
  CHECK_HYPER_FLOAT(update_model_ratio)
  CHECK_HYPER_UINT64(update_model_time_window)
  CHECK_HYPER_UINT64(client_epoch_num)
  CHECK_HYPER_UINT64(client_batch_size)
  CHECK_HYPER_FLOAT(client_learning_rate)
  CHECK_HYPER_UINT64(fl_iteration_num)
  float update_model_ratio = obj_new["update_model_ratio"];
  if (update_model_ratio <= 0.0 || update_model_ratio > 1.0) {
    *error_msg = "Field update_model_ratio in hyper param config should be in range of (0,1.0]";
    return false;
  }
  float client_learning_rate = obj_new["client_learning_rate"];
  if (client_learning_rate <= 0.0) {
    *error_msg = "Field client_learning_rate in hyper param config should be greater than 0";
    return false;
  }
  *output_hyper_params = obj_new.dump();
  return true;
}
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
