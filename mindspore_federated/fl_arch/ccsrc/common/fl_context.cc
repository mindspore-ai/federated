/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "common/fl_context.h"
#include <unordered_map>
#include "common/utils/log_adapter.h"

namespace mindspore {
namespace fl {
std::shared_ptr<FLContext> FLContext::instance() {
  static std::shared_ptr<FLContext> instance = nullptr;
  if (instance == nullptr) {
    instance.reset(new FLContext());
  }
  return instance;
}

void FLContext::LoadYamlConfig(const std::unordered_map<std::string, yaml::YamlConfigItem> &yaml_configs,
                               const std::string &yaml_config_file, const std::string &role, bool enable_ssl) {
  yaml::YamlConfig config;
  config.Load(yaml_configs, yaml_config_file, role, enable_ssl);
}

void FLContext::Reset() {}

std::string FLContext::ms_role() const { return role_; }

bool FLContext::is_worker() const { return role_ == kEnvRoleOfWorker; }

bool FLContext::is_server() const { return role_ == kEnvRoleOfServer; }

bool FLContext::is_scheduler() const { return role_ == kEnvRoleOfScheduler; }

void FLContext::set_server_mode(const std::string &server_mode) {
  if (server_mode != kServerModeFL && server_mode != kServerModeHybrid) {
    MS_LOG(EXCEPTION) << server_mode << " is invalid. Server mode must be " << kServerModeFL << " or "
                      << kServerModeHybrid;
    return;
  }
  MS_LOG(INFO) << "Server mode: " << server_mode << " is used for Server and Worker. Scheduler will ignore it.";
  server_mode_ = server_mode;
}

const std::string &FLContext::server_mode() const { return server_mode_; }

const std::string &FLContext::encrypt_type() const { return encrypt_config_.encrypt_type; }

void FLContext::set_ms_role(const std::string &role) {
  if (role != kEnvRoleOfWorker && role != kEnvRoleOfScheduler && role != kEnvRoleOfServer) {
    MS_LOG(EXCEPTION) << "ms_role " << role << " is invalid.";
    return;
  }
  MS_LOG(INFO) << "MS_ROLE of this node is " << role;
  role_ = role;
}

void FLContext::GenerateResetterRound() {
  uint32_t binary_server_context = 0;
  bool is_federated_learning_mode = false;
  bool is_mixed_training_mode = false;
  bool is_pairwise_encrypt = (encrypt_type() == kPWEncryptType);

  if (server_mode_ == kServerModeFL) {
    is_federated_learning_mode = true;
  } else if (server_mode_ == kServerModeHybrid) {
    is_mixed_training_mode = true;
  } else {
    MS_LOG(EXCEPTION) << server_mode_ << " is invalid. Server mode must be "
                      << " or " << kServerModeFL << " or " << kServerModeHybrid;
    return;
  }
  const int training_mode_offset = 2;
  const int pairwise_encrypt_offset = 3;
  binary_server_context = ((unsigned int)is_federated_learning_mode << 1) |
                          ((unsigned int)is_mixed_training_mode << training_mode_offset) |
                          ((unsigned int)is_pairwise_encrypt << pairwise_encrypt_offset);
  if (kServerContextToResetRoundMap.count(binary_server_context) == 0) {
    resetter_round_ = ResetterRound::kNoNeedToReset;
  } else {
    resetter_round_ = kServerContextToResetRoundMap.at(binary_server_context);
  }
  MS_LOG(INFO) << "Server context is " << binary_server_context << ". Resetter round is " << resetter_round_;
  return;
}

ResetterRound FLContext::resetter_round() const { return resetter_round_; }

void FLContext::set_http_server_address(const std::string &http_server_address) {
  http_server_address_ = http_server_address;
}

std::string FLContext::http_server_address() const { return http_server_address_; }

void FLContext::set_start_fl_job_threshold(uint64_t start_fl_job_threshold) {
  start_fl_job_threshold_ = start_fl_job_threshold;
}

uint64_t FLContext::start_fl_job_threshold() const { return start_fl_job_threshold_; }

void FLContext::set_start_fl_job_time_window(uint64_t start_fl_job_time_window) {
  start_fl_job_time_window_ = start_fl_job_time_window;
}

uint64_t FLContext::start_fl_job_time_window() const { return start_fl_job_time_window_; }

void FLContext::set_update_model_ratio(float update_model_ratio) {
  if (update_model_ratio > 1.0 || update_model_ratio <= 0) {
    MS_LOG(EXCEPTION) << "update_model_ratio must be in range (0, 1.0]";
    return;
  }
  update_model_ratio_ = update_model_ratio;
}

float FLContext::update_model_ratio() const { return update_model_ratio_; }

void FLContext::set_update_model_time_window(uint64_t update_model_time_window) {
  update_model_time_window_ = update_model_time_window;
}

uint64_t FLContext::update_model_time_window() const { return update_model_time_window_; }

void FLContext::set_fl_name(const std::string &fl_name) { fl_name_ = fl_name; }

const std::string &FLContext::fl_name() const { return fl_name_; }

void FLContext::set_fl_iteration_num(uint64_t fl_iteration_num) { fl_iteration_num_ = fl_iteration_num; }

uint64_t FLContext::fl_iteration_num() const { return fl_iteration_num_; }

void FLContext::set_ssl_config(const SslConfig &config) { ssl_config_ = config; }

const SslConfig &FLContext::ssl_config() const { return ssl_config_; }

void FLContext::set_client_verify_config(const ClientVerifyConfig &config) { client_verify_config_ = config; }

const ClientVerifyConfig &FLContext::client_verify_config() const { return client_verify_config_; }

void FLContext::set_distributed_cache_config(const DistributedCacheConfig &config) {
  distributed_cache_config_ = config;
}

const DistributedCacheConfig &FLContext::distributed_cache_config() const { return distributed_cache_config_; }

void FLContext::set_encrypt_config(const EncryptConfig &config) {
  encrypt_config_ = config;
  auto &encrypt_type = encrypt_config_.encrypt_type;
  if (encrypt_type != kNotEncryptType && encrypt_type != kDPEncryptType && encrypt_type != kPWEncryptType &&
      encrypt_type != kStablePWEncryptType && encrypt_type != kDSEncryptType) {
    MS_LOG(EXCEPTION) << encrypt_type << " is invalid. Encrypt type must be " << kNotEncryptType << " or "
                      << kDPEncryptType << " or " << kPWEncryptType << " or " << kStablePWEncryptType << " or "
                      << kDSEncryptType;
  }
  if (encrypt_type == kNotEncryptType) {
    return;
  }
  CheckDPEncrypt(config);
  CheckSignDsEncrypt(config);
  CheckPWEncrypt(config);
}
void FLContext::CheckPWEncrypt(const EncryptConfig &config) const {
  if (config.share_secrets_ratio <= 0 || config.share_secrets_ratio > 1) {
    MS_LOG(EXCEPTION) << config.share_secrets_ratio << " is invalid, share_secrets_ratio must be in range of (0, 1].";
  }

  if (config.cipher_time_window < 0) {
    MS_LOG(EXCEPTION) << "cipher_time_window " << config.share_secrets_ratio << " should not be less than 0.";
  }
  if (config.reconstruct_secrets_threshold == 0) {
    MS_LOG(EXCEPTION) << "reconstruct_secrets_threshold should be positive.";
  }
}

void FLContext::CheckSignDsEncrypt(const EncryptConfig &config) const {
  const float sign_k_upper = 0.25;
  if (config.sign_k <= 0 || config.sign_k > sign_k_upper) {
    MS_LOG(EXCEPTION) << config.sign_k << " is invalid, sign_k must be in range of (0, 0.25], 0.01 is used by default.";
  }
  const float sign_eps_upper = 100;
  if (config.sign_eps <= 0 || config.sign_eps > sign_eps_upper) {
    MS_LOG(EXCEPTION) << config.sign_eps
                      << " is invalid, sign_eps must be in range of (0, 100], 100 is used by default.";
  }
  const float sign_thr_ratio_bound = 0.5;
  if (config.sign_thr_ratio < sign_thr_ratio_bound || config.sign_thr_ratio > 1) {
    MS_LOG(EXCEPTION) << config.sign_thr_ratio
                      << " is invalid, sign_thr_ratio must be in range of [0.5, 1], 0.6 is used by default.";
  }
  if (config.sign_global_lr <= 0) {
    MS_LOG(EXCEPTION) << config.sign_global_lr
                      << " is invalid, sign_global_lr must be larger than 0, 1 is used by default.";
  }
  const int sign_dim_out_upper = 50;
  if (config.sign_dim_out < 0 || config.sign_dim_out > sign_dim_out_upper) {
    MS_LOG(EXCEPTION) << config.sign_dim_out
                      << " is invalid, sign_dim_out must be in range of [0, 50], 0 is used by default.";
  }
}

void FLContext::CheckDPEncrypt(const EncryptConfig &config) const {
  if (config.dp_eps <= 0) {
    MS_LOG(EXCEPTION) << config.dp_eps << " is invalid, dp_eps must be larger than 0, 50 is used by default.";
  }
  if (config.dp_delta <= 0 || config.dp_delta >= 1) {
    MS_LOG(EXCEPTION) << config.dp_delta
                      << " is invalid, dp_delta must be in range of (0, 1), 0.01 is used by default.";
  }
  if (config.dp_norm_clip <= 0) {
    MS_LOG(EXCEPTION) << config.dp_norm_clip
                      << " is invalid, dp_norm_clip must be larger than 0, 1 is used by default.";
  }
}

const EncryptConfig &FLContext::encrypt_config() const { return encrypt_config_; }

void FLContext::set_compression_config(const CompressionConfig &config) { compression_config_ = config; }

const CompressionConfig &FLContext::compression_config() const { return compression_config_; }

void FLContext::set_client_epoch_num(uint64_t client_epoch_num) { client_epoch_num_ = client_epoch_num; }

uint64_t FLContext::client_epoch_num() const { return client_epoch_num_; }

void FLContext::set_client_batch_size(uint64_t client_batch_size) { client_batch_size_ = client_batch_size; }

uint64_t FLContext::client_batch_size() const { return client_batch_size_; }

void FLContext::set_client_learning_rate(float client_learning_rate) { client_learning_rate_ = client_learning_rate; }

float FLContext::client_learning_rate() const { return client_learning_rate_; }

void FLContext::set_max_connection_num(uint64_t max_connection_num) { max_collection_num_ = max_connection_num; }

uint64_t FLContext::max_connection_num() const { return max_collection_num_; }

void FLContext::set_secure_aggregation(bool secure_aggregation) { secure_aggregation_ = secure_aggregation; }

bool FLContext::secure_aggregation() const { return secure_aggregation_; }

ClusterConfig &FLContext::cluster_config() {
  if (cluster_config_ == nullptr) {
    cluster_config_ = std::make_unique<ClusterConfig>();
  }
  return *cluster_config_;
}

bool FLContext::pki_verify() const { return client_verify_config_.pki_verify; }

void FLContext::set_scheduler_manage_address(const std::string &manage_address) {
  scheduler_manage_address_ = manage_address;
}

std::string FLContext::scheduler_manage_address() const { return scheduler_manage_address_; }

bool FLContext::enable_ssl() const { return enable_ssl_; }

void FLContext::set_enable_ssl(bool enabled) { enable_ssl_ = enabled; }

std::string FLContext::client_password() const { return client_password_; }
void FLContext::set_client_password(const std::string &password) { client_password_ = password; }

std::string FLContext::server_password() const { return server_password_; }
void FLContext::set_server_password(const std::string &password) { server_password_ = password; }

std::string FLContext::http_url_prefix() const { return http_url_prefix_; }

void FLContext::set_http_url_prefix(const std::string &http_url_prefix) { http_url_prefix_ = http_url_prefix; }

void FLContext::set_global_iteration_time_window(const uint64_t &global_iteration_time_window) {
  global_iteration_time_window_ = global_iteration_time_window;
}

uint64_t FLContext::global_iteration_time_window() const { return global_iteration_time_window_; }

std::string FLContext::checkpoint_dir() const { return checkpoint_dir_; }

void FLContext::set_checkpoint_dir(const std::string &checkpoint_dir) { checkpoint_dir_ = checkpoint_dir; }

void FLContext::set_instance_name(const std::string &instance_name) { instance_name_ = instance_name; }

const std::string &FLContext::instance_name() const { return instance_name_; }

void FLContext::set_participation_time_level(const std::string &participation_time_level) {
  participation_time_level_ = participation_time_level;
}

const std::string &FLContext::participation_time_level() { return participation_time_level_; }

void FLContext::set_continuous_failure_times(uint32_t continuous_failure_times) {
  continuous_failure_times_ = continuous_failure_times;
}

uint32_t FLContext::continuous_failure_times() const { return continuous_failure_times_; }

void FLContext::set_metrics_file(const std::string &metrics_file) { metrics_file_ = metrics_file; }
const std::string &FLContext::metrics_file() { return metrics_file_; }

void FLContext::set_failure_event_file(const std::string &failure_event_file) {
  failure_event_file_ = failure_event_file;
}
const std::string &FLContext::failure_event_file() { return failure_event_file_; }

void FLContext::set_data_rate_dir(const std::string &data_rate_dir) { data_rate_dir_ = data_rate_dir; }
const std::string &FLContext::data_rate_dir() { return data_rate_dir_; }

void FLContext::set_tcp_server_ip(const std::string &tcp_server_ip) { tcp_server_ip_ = tcp_server_ip; }
std::string FLContext::tcp_server_ip() const { return tcp_server_ip_; }
}  // namespace fl
}  // namespace mindspore
