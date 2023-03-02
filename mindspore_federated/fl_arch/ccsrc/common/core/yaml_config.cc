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
#include "common/core/yaml_config.h"
#include <functional>
#include <vector>
#include "common/common.h"
#include "common/fl_context.h"

namespace mindspore {
namespace fl {
namespace yaml {
void YamlConfig::Load(const std::unordered_map<std::string, YamlConfigItem> &items, const std::string &yaml_config_file,
                      const std::string &role) {
  items_ = items;
  yaml_config_file_ = yaml_config_file;
  InitCommonConfig();
  if (role == kEnvRoleOfServer) {
    InitRoundConfig();
    InitSummaryConfig();
    InitUnsupervisedConfig();
    InitEncryptConfig();
    InitCompressionConfig();
    InitClientVerifyConfig();
    InitClientConfig();
    CheckYamlConfig();
  }
}

#define SET_INT_CXT(func) [](uint64_t val) { FLContext::instance()->func(val); }
#define SET_FLOAT_CXT(func) [](float val) { FLContext::instance()->func(val); }
#define SET_BOOL_CXT(func) [](bool val) { FLContext::instance()->func(val); }
#define SET_STR_CXT(func) [](const std::string &val) { FLContext::instance()->func(val); }

void YamlConfig::InitCommonConfig() {
  Get("fl_name", SET_STR_CXT(set_fl_name), true);
  Get("fl_iteration_num", SET_INT_CXT(set_fl_iteration_num), true, CheckInt(1, UINT32_MAX, INC_BOTH));
  Get("server_mode", SET_STR_CXT(set_server_mode), true);
  Get("enable_ssl", SET_BOOL_CXT(set_enable_ssl), true);
  // multi aggregation algorithm
  InitAggregationConfig();
  // distributed cache
  InitDistributedCacheConfig();
  if (FLContext::instance()->enable_ssl()) {
    InitSslConfig();
  }
}

void YamlConfig::InitAggregationConfig() {
  AggregationConfig aggregation_config;
  Get("aggregation.aggregation_type", &aggregation_config.aggregation_type, false,
      {kFedAvgAggregation, kFedProxAggregation, kScaffoldAggregation, kFedNovaAggregation});
  Get("aggregation.iid_rate", &aggregation_config.iid_rate, false, CheckFloat(0, 1, INC_RIGHT));
  Get("aggregation.total_client_num", &aggregation_config.total_client_num, false, CheckInt(1, UINT32_MAX, INC_BOTH));
  FLContext::instance()->set_aggregation_config(aggregation_config);
}

void YamlConfig::InitDistributedCacheConfig() {
  DistributedCacheConfig distributed_cache_config;
  Get("distributed_cache.type", &distributed_cache_config.type, false);
  Get("distributed_cache.address", &distributed_cache_config.address, true);
  Get("distributed_cache.plugin_lib_path", &distributed_cache_config.plugin_lib_path, false);
  std::string prefix = "distributed_cache.";
  for (const auto &item : items_) {
    if (item.first.length() <= prefix.length() || item.first.substr(0, prefix.length()) != prefix) {
      continue;
    }
    auto key = item.first.substr(prefix.length());
    std::string val;
    switch (item.second.type) {
      case kYamlInt:
        val = std::to_string(item.second.int_val);
        break;
      case kYamlFloat:
        val = std::to_string(item.second.float_val);
        break;
      case kYamlBool:
        val = std::to_string(item.second.bool_val);
        break;
      case kYamlStr:
        val = item.second.str_val;
        break;
      default:
        continue;
    }
    distributed_cache_config.configs[key] = val;
  }
  FLContext::instance()->set_distributed_cache_config(distributed_cache_config);
}

void YamlConfig::InitSslConfig() {
  SslConfig ssl_config;
  // for tcp/http server
  Get("ssl.server_cert_path", &ssl_config.server_cert_path, true);
  // for tcp client
  Get("ssl.client_cert_path", &ssl_config.client_cert_path, true);
  // common
  Get("ssl.ca_cert_path", &ssl_config.ca_cert_path, true);
  Get("ssl.crl_path", &ssl_config.crl_path, false);
  Get("ssl.cipher_list", &ssl_config.cipher_list, true);
  Get("ssl.cert_expire_warning_time_in_day", &ssl_config.cert_expire_warning_time_in_day, false,
      CheckInt(kMinWarningTime, kMaxWarningTime, INC_BOTH));
  FLContext::instance()->set_ssl_config(ssl_config);
}

void YamlConfig::InitRoundConfig() {
  Get("round.start_fl_job_threshold", SET_INT_CXT(set_start_fl_job_threshold), true, CheckInt(1, UINT32_MAX, INC_BOTH));
  Get("round.start_fl_job_time_window", SET_INT_CXT(set_start_fl_job_time_window), true,
      CheckInt(1, UINT32_MAX, INC_BOTH));
  Get("round.update_model_ratio", SET_FLOAT_CXT(set_update_model_ratio), true,
      CheckFloat(0, 1, INC_RIGHT));  // (0, 1.0]
  Get("round.update_model_time_window", SET_INT_CXT(set_update_model_time_window), true,
      CheckInt(1, UINT32_MAX, INC_BOTH));
  Get("round.global_iteration_time_window", SET_INT_CXT(set_global_iteration_time_window), true,
      CheckInt(1, UINT32_MAX, INC_BOTH));
}

void YamlConfig::InitSummaryConfig() {
  Get("summary.participation_time_level", SET_STR_CXT(set_participation_time_level), false);
  Get("summary.continuous_failure_times", SET_INT_CXT(set_continuous_failure_times), false, CheckInt(1, GE));
  Get("summary.metrics_file", SET_STR_CXT(set_metrics_file), true);
  Get("summary.failure_event_file", SET_STR_CXT(set_failure_event_file), true);
  Get("summary.data_rate_dir", SET_STR_CXT(set_data_rate_dir), false);
}

void YamlConfig::InitUnsupervisedConfig() {
  UnsupervisedConfig unsupervised_config;
  Get("unsupervised.cluster_client_num", &unsupervised_config.cluster_client_num, false, CheckInt(1, GE));
  Get("unsupervised.eval_type", &unsupervised_config.eval_type, false,
      {kNotEvalType, kSilhouetteScoreType, kCalinskiHarabaszScoreType});

  if (unsupervised_config.eval_type != kNotEvalType && unsupervised_config.cluster_client_num <= 0) {
    MS_LOG(EXCEPTION) << "Cluster client num is <= 0 when unsupervised eval mode is opened.";
  }

  MS_LOG(INFO) << "cluster_client_num is " << unsupervised_config.cluster_client_num << ", eval_type is "
               << unsupervised_config.eval_type;
  FLContext::instance()->set_unsupervised_config(unsupervised_config);
}

void YamlConfig::InitEncryptConfig() {
  EncryptConfig encrypt_config;
  Get("encrypt.encrypt_type", &encrypt_config.encrypt_type, false,
      {kNotEncryptType, kPWEncryptType, kStablePWEncryptType, kDPEncryptType, kDSEncryptType});
  auto &encrypt_type = encrypt_config.encrypt_type;
  if (encrypt_type == kPWEncryptType || encrypt_type == kStablePWEncryptType) {
    Get("encrypt.pw_encrypt.share_secrets_ratio", &encrypt_config.share_secrets_ratio, true,
        CheckFloat(0, 1, INC_RIGHT));  // (0, 1.0]
    Get("encrypt.pw_encrypt.cipher_time_window", &encrypt_config.cipher_time_window, true,
        CheckInt(1, UINT32_MAX, INC_BOTH));
    Get("encrypt.pw_encrypt.reconstruct_secrets_threshold", &encrypt_config.reconstruct_secrets_threshold, true,
        CheckInt(1, UINT32_MAX, INC_BOTH));
  } else if (encrypt_type == kDPEncryptType) {
    Get("encrypt.dp_encrypt.dp_eps", &encrypt_config.dp_eps, false, CheckFloat(0, GT));                  // >0
    Get("encrypt.dp_encrypt.dp_delta", &encrypt_config.dp_delta, false, CheckFloat(0, 1, INC_NEITHER));  // (0,1)
    Get("encrypt.dp_encrypt.dp_norm_clip", &encrypt_config.dp_norm_clip, false, CheckFloat(0, GT));      // >0
  } else if (encrypt_type == kDSEncryptType) {
    Get("encrypt.signds.sign_k", &encrypt_config.sign_k, false, CheckFloat(0, 0.25, INC_RIGHT));     // (0, 0.25]
    Get("encrypt.signds.sign_eps", &encrypt_config.sign_eps, false, CheckFloat(0, 100, INC_RIGHT));  // (0, 100]
    Get("encrypt.signds.sign_thr_ratio", &encrypt_config.sign_thr_ratio, false,
        CheckFloat(0.5, 1, INC_BOTH));                                                                   // [0.5, 1]
    Get("encrypt.signds.sign_global_lr", &encrypt_config.sign_global_lr, false, CheckFloat(0, GT));      // >0
    Get("encrypt.signds.sign_dim_out", &encrypt_config.sign_dim_out, false, CheckInt(0, 50, INC_BOTH));  // [0, 50]
  }
  FLContext::instance()->set_encrypt_config(encrypt_config);
}

void YamlConfig::InitCompressionConfig() {
  CompressionConfig compression_config;
  Get("compression.upload_compress_type", &compression_config.upload_compress_type, false,
      {kNoCompressType, kDiffSparseQuant});
  Get("compression.upload_sparse_rate", &compression_config.upload_sparse_rate, false, CheckFloat(0, 1, INC_RIGHT));
  Get("compression.download_compress_type", &compression_config.download_compress_type, false,
      {kNoCompressType, kQuant});
  FLContext::instance()->set_compression_config(compression_config);

  MS_LOG(INFO) << "upload_compress_type is " << compression_config.upload_compress_type << ", upload_sparse_rate is "
               << compression_config.upload_sparse_rate << ", download_compress_type is "
               << compression_config.download_compress_type;
}

void YamlConfig::InitClientVerifyConfig() {
  ClientVerifyConfig http_config;
  Get("client_verify.pki_verify", &http_config.pki_verify, false);
  Get("client_verify.root_first_ca_path", &http_config.root_first_ca_path, false);
  Get("client_verify.root_second_ca_path", &http_config.root_second_ca_path, false);
  Get("client_verify.equip_crl_path", &http_config.equip_crl_path, false);
  Get("client_verify.replay_attack_time_diff", &http_config.replay_attack_time_diff, false, CheckInt(0, GE));
  FLContext::instance()->set_client_verify_config(http_config);
}

void YamlConfig::InitClientConfig() {
  Get("client.http_url_prefix", SET_STR_CXT(set_http_url_prefix), false);
  Get("client.client_epoch_num", SET_INT_CXT(set_client_epoch_num), false, CheckInt(1, GE));
  Get("client.client_batch_size", SET_INT_CXT(set_client_batch_size), false, CheckInt(1, GE));
  Get("client.client_learning_rate", SET_FLOAT_CXT(set_client_learning_rate), false, CheckFloat(0, GT));
  Get("client.connection_num", SET_INT_CXT(set_max_connection_num), false, CheckInt(1, GE));
}

void YamlConfig::CheckYamlConfig() {
  CompressionConfig compression_config = FLContext::instance()->compression_config();
  auto upload_compress_type = compression_config.upload_compress_type;
  auto encrypt_type = FLContext::instance()->encrypt_config().encrypt_type;
  if (upload_compress_type != kNoCompressType && (encrypt_type == kDSEncryptType || encrypt_type == kPWEncryptType)) {
    MS_LOG(WARNING) << "The '" << encrypt_type << "' and '{" << upload_compress_type << "}' are conflicted, and in '{"
                    << encrypt_type << "}' mode the 'upload_compress_type' will be 'NO_COMPRESS'";
    compression_config.upload_compress_type = kNoCompressType;
    FLContext::instance()->set_compression_config(compression_config);
  }
}

LogStream &operator<<(LogStream &os, YamlValType type) {
  std::unordered_map<YamlValType, std::string> type_str = {
    {kYamlInt, "int"}, {kYamlFloat, "float"}, {kYamlBool, "bool"}, {kYamlStr, "str"}, {kYamlDict, "dict"}};
  auto it = type_str.find(type);
  if (it == type_str.end()) {
    os << "[invalid type]";
  } else {
    os << it->second;
  }
  return os;
}

void YamlConfig::Get(const std::string &key, const std::function<void(const std::string &)> &set_fun, bool required,
                     const std::vector<std::string> &choices) const {
  if (set_fun == nullptr) {
    return;
  }
  std::string val;
  if (!Get(key, &val, required, choices)) {
    return;
  }
  set_fun(val);
}

void YamlConfig::Get(const std::string &key, const std::function<void(uint64_t)> &set_fun, bool required,
                     CheckNum<int64_t> check) const {
  if (set_fun == nullptr) {
    return;
  }
  uint64_t val = 0;
  if (!Get(key, &val, required, check)) {
    return;
  }
  set_fun(val);
}

void YamlConfig::Get(const std::string &key, const std::function<void(float)> &set_fun, bool required,
                     CheckNum<float> check) const {
  if (set_fun == nullptr) {
    return;
  }
  float val = 0.0f;
  if (!Get(key, &val, required, check)) {
    return;
  }
  set_fun(val);
}

void YamlConfig::Get(const std::string &key, const std::function<void(bool)> &set_fun, bool required) const {
  if (set_fun == nullptr) {
    return;
  }
  bool val = false;
  if (!Get(key, &val, required)) {
    return;
  }
  set_fun(val);
}

bool YamlConfig::Get(const std::string &key, std::string *value, bool required,
                     const std::vector<std::string> &choices) const {
  auto it = items_.find(key);
  if (it == items_.end()) {
    if (required) {
      MS_LOG_EXCEPTION << "The parameter '" << key << "' is missing, yaml config file: " << yaml_config_file_;
    }
    return false;
  }
  auto &item = it->second;
  if (item.type != kYamlStr) {
    MS_LOG_EXCEPTION << "The parameter '" << key << "' is expected to be type str, actually << " << item.type
                     << ", yaml config file: " << yaml_config_file_;
  }
  if (item.str_val.empty()) {
    if (required) {
      MS_LOG_EXCEPTION << "The parameter '" << key << "' is missing, yaml config file: " << yaml_config_file_;
    }
    return false;
  }
  if (!choices.empty() && std::find(choices.begin(), choices.end(), item.str_val) == choices.end()) {
    MS_LOG_EXCEPTION << "The value of parameter '" << key << "' can be only one of " << choices << ", but got "
                     << item.str_val << ", yaml config file: " << yaml_config_file_;
  }
  *value = item.str_val;
  return true;
}

bool YamlConfig::Get(const std::string &key, uint64_t *value, bool required, CheckInt check) const {
  auto it = items_.find(key);
  if (it == items_.end()) {
    if (required) {
      MS_LOG_EXCEPTION << "The parameter '" << key << "' is missing, yaml config file: " << yaml_config_file_;
    }
    return false;
  }
  auto &item = it->second;
  if (item.type != kYamlInt) {
    MS_LOG_EXCEPTION << "The parameter '" << key << "' is expected to be type int, actually " << item.type
                     << ", yaml config file: " << yaml_config_file_;
  }
  auto check_ret = check.Check(item.int_val);
  if (!check_ret.IsSuccess()) {
    MS_LOG_EXCEPTION << "Failed to check value of parameter '" << key << "': " << check_ret.StatusMessage()
                     << ", yaml config file: " << yaml_config_file_;
  }
  *value = static_cast<uint64_t>(item.int_val);
  return true;
}

bool YamlConfig::Get(const std::string &key, float *value, bool required, CheckFloat check) const {
  auto it = items_.find(key);
  if (it == items_.end()) {
    if (required) {
      MS_LOG_EXCEPTION << "The parameter '" << key << "' is missing, yaml config file: " << yaml_config_file_;
    }
    return false;
  }
  auto &item = it->second;
  if (item.type == kYamlFloat) {
    *value = item.float_val;
  } else if (item.type == kYamlInt) {
    *value = static_cast<float>(item.int_val);
  } else {
    MS_LOG_EXCEPTION << "The parameter '" << key << "' is expected to be type float, actually " << item.type
                     << ", yaml config file: " << yaml_config_file_;
  }
  auto check_ret = check.Check(*value);
  if (!check_ret.IsSuccess()) {
    MS_LOG_EXCEPTION << "Failed to check value of parameter '" << key << "': " << check_ret.StatusMessage()
                     << ", yaml config file: " << yaml_config_file_;
  }
  return true;
}

bool YamlConfig::Get(const std::string &key, bool *value, bool required) const {
  auto it = items_.find(key);
  if (it == items_.end()) {
    if (required) {
      MS_LOG_EXCEPTION << "The parameter '" << key << "' is missing, yaml config file: " << yaml_config_file_;
    }
    return false;
  }
  auto &item = it->second;
  if (item.type != kYamlBool) {
    MS_LOG_EXCEPTION << "The parameter '" << key << "' is expected to be type bool, actually " << item.type
                     << ", yaml config file: " << yaml_config_file_;
  }
  *value = item.bool_val;
  return true;
}
}  // namespace yaml
}  // namespace fl
}  // namespace mindspore
