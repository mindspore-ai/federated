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

#include "python/fl_context.h"
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

void FLContext::Reset() {
  is_worker_ = false;
  is_server_ = false;
  is_sched_ = false;
}

std::string FLContext::ms_role() const {
  if (is_worker_) {
    return kEnvRoleOfWorker;
  } else if (is_server_) {
    return kEnvRoleOfServer;
  } else if (is_sched_) {
    return kEnvRoleOfScheduler;
  } else {
    MS_LOG(EXCEPTION) << "MS role is disabled.";
  }
}

bool FLContext::is_worker() const { return is_worker_; }

bool FLContext::is_server() const { return is_server_; }

bool FLContext::is_scheduler() const { return is_sched_; }

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

void FLContext::set_encrypt_type(const std::string &encrypt_type) {
  if (encrypt_type != kNotEncryptType && encrypt_type != kDPEncryptType && encrypt_type != kPWEncryptType &&
      encrypt_type != kStablePWEncryptType && encrypt_type != kDSEncryptType) {
    MS_LOG(WARNING) << encrypt_type << " is invalid. Encrypt type must be " << kNotEncryptType << " or "
                    << kDPEncryptType << " or " << kPWEncryptType << " or " << kStablePWEncryptType << " or "
                    << kDSEncryptType << ", DP scheme is used by default.";
    encrypt_type_ = kDPEncryptType;
  } else {
    encrypt_type_ = encrypt_type;
  }
}

const std::string &FLContext::encrypt_type() const { return encrypt_type_; }

void FLContext::set_dp_eps(float dp_eps) {
  if (dp_eps > 0) {
    dp_eps_ = dp_eps;
  } else {
    MS_LOG(WARNING) << dp_eps << " is invalid, dp_eps must be larger than 0, 50 is used by default.";
    float dp_eps_default = 50;
    dp_eps_ = dp_eps_default;
  }
}

float FLContext::dp_eps() const { return dp_eps_; }

void FLContext::set_dp_delta(float dp_delta) {
  if (dp_delta > 0 && dp_delta < 1) {
    dp_delta_ = dp_delta;
  } else {
    MS_LOG(WARNING) << dp_delta << " is invalid, dp_delta must be in range of (0, 1), 0.01 is used by default.";
    float dp_delta_default = 0.01;
    dp_delta_ = dp_delta_default;
  }
}
float FLContext::dp_delta() const { return dp_delta_; }

void FLContext::set_dp_norm_clip(float dp_norm_clip) {
  if (dp_norm_clip > 0) {
    dp_norm_clip_ = dp_norm_clip;
  } else {
    MS_LOG(WARNING) << dp_norm_clip << " is invalid, dp_norm_clip must be larger than 0, 1 is used by default.";
    float dp_norm_clip_default = 1;
    dp_norm_clip_ = dp_norm_clip_default;
  }
}
float FLContext::dp_norm_clip() const { return dp_norm_clip_; }

void FLContext::set_sign_k(float sign_k) {
  float sign_k_upper = 0.25;
  if (sign_k > 0 && sign_k <= sign_k_upper) {
    sign_k_ = sign_k;
  } else {
    MS_LOG(WARNING) << sign_k << " is invalid, sign_k must be in range of (0, 0.25], 0.01 is used by default.";
    float sign_k_default = 0.01;
    sign_k_ = sign_k_default;
  }
}
float FLContext::sign_k() const { return sign_k_; }

void FLContext::set_sign_eps(float sign_eps) {
  float sign_eps_upper = 100;
  if (sign_eps > 0 && sign_eps <= sign_eps_upper) {
    sign_eps_ = sign_eps;
  } else {
    MS_LOG(WARNING) << sign_eps << " is invalid, sign_eps must be in range of (0, 100], 100 is used by default.";
    float sign_eps_default = 100;
    sign_eps_ = sign_eps_default;
  }
}
float FLContext::sign_eps() const { return sign_eps_; }

void FLContext::set_sign_thr_ratio(float sign_thr_ratio) {
  float sign_thr_ratio_bound = 0.5;
  if (sign_thr_ratio >= sign_thr_ratio_bound && sign_thr_ratio <= 1) {
    sign_thr_ratio_ = sign_thr_ratio;
  } else {
    MS_LOG(WARNING) << sign_thr_ratio
                    << " is invalid, sign_thr_ratio must be in range of [0.5, 1], 0.6 is used by default.";
    float sign_thr_ratio_default = 0.6;
    sign_thr_ratio_ = sign_thr_ratio_default;
  }
}
float FLContext::sign_thr_ratio() const { return sign_thr_ratio_; }

void FLContext::set_sign_global_lr(float sign_global_lr) {
  if (sign_global_lr > 0) {
    sign_global_lr_ = sign_global_lr;
  } else {
    MS_LOG(WARNING) << sign_global_lr << " is invalid, sign_global_lr must be larger than 0, 1 is used by default.";
    float sign_global_lr_default = 1;
    sign_global_lr_ = sign_global_lr_default;
  }
}
float FLContext::sign_global_lr() const { return sign_global_lr_; }

void FLContext::set_sign_dim_out(int sign_dim_out) {
  int sign_dim_out_upper = 50;
  if (sign_dim_out >= 0 && sign_dim_out <= sign_dim_out_upper) {
    sign_dim_out_ = sign_dim_out;
  } else {
    MS_LOG(WARNING) << sign_dim_out << " is invalid, sign_dim_out must be in range of [0, 50], 0 is used by default.";
    sign_dim_out_ = 0;
  }
}
int FLContext::sign_dim_out() const { return sign_dim_out_; }

void FLContext::set_ms_role(const std::string &role) {
  if (role == kEnvRoleOfWorker) {
    is_worker_ = true;
  } else if (role == kEnvRoleOfServer) {
    is_server_ = true;
  } else if (role == kEnvRoleOfScheduler) {
    is_sched_ = true;
  } else {
    MS_LOG(EXCEPTION) << "ms_role " << role << " is invalid.";
    return;
  }
  MS_LOG(INFO) << "MS_ROLE of this node is " << role;
  role_ = role;
}

void FLContext::set_worker_num(uint32_t worker_num) {
  // Hybrid training mode only supports one worker for now.
  if (server_mode_ == kServerModeHybrid && worker_num != 1) {
    MS_LOG(EXCEPTION) << "The worker number should be set to 1 in hybrid training mode.";
    return;
  }
  worker_num_ = worker_num;
}
uint32_t FLContext::initial_worker_num() const { return worker_num_; }

void FLContext::set_server_num(uint32_t server_num) { server_num_ = server_num; }
uint32_t FLContext::initial_server_num() const { return server_num_; }

void FLContext::set_scheduler_ip(const std::string &sched_ip) {
  if (sched_ip.length() > kLength) {
    MS_LOG(EXCEPTION) << "The scheduler ip's length can not exceed " << kLength;
  }
  scheduler_ip_ = sched_ip;
}

std::string FLContext::scheduler_ip() const { return scheduler_ip_; }

void FLContext::set_scheduler_port(uint16_t sched_port) {
  if (sched_port > kMaxPort) {
    MS_LOG(EXCEPTION) << "The port: " << sched_port << " is illegal.";
  }
  scheduler_port_ = sched_port;
}

uint16_t FLContext::scheduler_port() const { return scheduler_port_; }

void FLContext::GenerateResetterRound() {
  uint32_t binary_server_context = 0;
  bool is_federated_learning_mode = false;
  bool is_mixed_training_mode = false;
  bool is_pairwise_encrypt = (encrypt_type_ == kPWEncryptType);

  if (server_mode_ == kServerModeFL) {
    is_federated_learning_mode = true;
  } else if (server_mode_ == kServerModeHybrid) {
    is_mixed_training_mode = true;
  } else {
    MS_LOG(EXCEPTION) << server_mode_ << " is invalid. Server mode must be " << " or " << kServerModeFL
                      << " or " << kServerModeHybrid;
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

void FLContext::set_fl_server_port(uint16_t fl_server_port) { fl_server_port_ = fl_server_port; }

uint16_t FLContext::fl_server_port() const { return fl_server_port_; }

void FLContext::set_fl_client_enable(bool enabled) { fl_client_enable_ = enabled; }

bool FLContext::fl_client_enable() const { return fl_client_enable_; }

void FLContext::set_start_fl_job_threshold(uint64_t start_fl_job_threshold) {
  start_fl_job_threshold_ = start_fl_job_threshold;
}

uint64_t FLContext::start_fl_job_threshold() const { return start_fl_job_threshold_; }

void FLContext::set_start_fl_job_time_window(uint64_t start_fl_job_time_window) {
  start_fl_job_time_window_ = start_fl_job_time_window;
}

uint64_t FLContext::start_fl_job_time_window() const { return start_fl_job_time_window_; }

void FLContext::set_update_model_ratio(float update_model_ratio) {
  if (update_model_ratio > 1.0) {
    MS_LOG(EXCEPTION) << "update_model_ratio must be between 0 and 1.";
    return;
  }
  update_model_ratio_ = update_model_ratio;
}

float FLContext::update_model_ratio() const { return update_model_ratio_; }

void FLContext::set_update_model_time_window(uint64_t update_model_time_window) {
  update_model_time_window_ = update_model_time_window;
}

uint64_t FLContext::update_model_time_window() const { return update_model_time_window_; }

void FLContext::set_share_secrets_ratio(float share_secrets_ratio) {
  if (share_secrets_ratio > 0 && share_secrets_ratio <= 1) {
    share_secrets_ratio_ = share_secrets_ratio;
  } else {
    MS_LOG(EXCEPTION) << share_secrets_ratio << " is invalid, share_secrets_ratio must be in range of (0, 1].";
    return;
  }
}

float FLContext::share_secrets_ratio() const { return share_secrets_ratio_; }

void FLContext::set_cipher_time_window(uint64_t cipher_time_window) {
  if (cipher_time_window_ < 0) {
    MS_LOG(EXCEPTION) << "cipher_time_window should not be less than 0.";
    return;
  }
  cipher_time_window_ = cipher_time_window;
}

uint64_t FLContext::cipher_time_window() const { return cipher_time_window_; }

void FLContext::set_reconstruct_secrets_threshold(uint64_t reconstruct_secrets_threshold) {
  if (reconstruct_secrets_threshold == 0) {
    MS_LOG(EXCEPTION) << "reconstruct_secrets_threshold should be positive.";
    return;
  }
  reconstruct_secrets_threshold_ = reconstruct_secrets_threshold;
}

uint64_t FLContext::reconstruct_secrets_threshold() const { return reconstruct_secrets_threshold_; }

void FLContext::set_fl_name(const std::string &fl_name) { fl_name_ = fl_name; }

const std::string &FLContext::fl_name() const { return fl_name_; }

void FLContext::set_fl_iteration_num(uint64_t fl_iteration_num) { fl_iteration_num_ = fl_iteration_num; }

uint64_t FLContext::fl_iteration_num() const { return fl_iteration_num_; }

void FLContext::set_client_epoch_num(uint64_t client_epoch_num) { client_epoch_num_ = client_epoch_num; }

uint64_t FLContext::client_epoch_num() const { return client_epoch_num_; }

void FLContext::set_client_batch_size(uint64_t client_batch_size) { client_batch_size_ = client_batch_size; }

uint64_t FLContext::client_batch_size() const { return client_batch_size_; }

void FLContext::set_client_learning_rate(float client_learning_rate) { client_learning_rate_ = client_learning_rate; }

float FLContext::client_learning_rate() const { return client_learning_rate_; }

void FLContext::set_secure_aggregation(bool secure_aggregation) { secure_aggregation_ = secure_aggregation; }

bool FLContext::secure_aggregation() const { return secure_aggregation_; }

fl::core::ClusterConfig &FLContext::cluster_config() {
  if (cluster_config_ == nullptr) {
    cluster_config_ = std::make_unique<fl::core::ClusterConfig>(worker_num_, server_num_, scheduler_ip_, scheduler_port_);
  }
  return *cluster_config_;
}

void FLContext::set_root_first_ca_path(const std::string &root_first_ca_path) {
  root_first_ca_path_ = root_first_ca_path;
}
void FLContext::set_root_second_ca_path(const std::string &root_second_ca_path) {
  root_second_ca_path_ = root_second_ca_path;
}

std::string FLContext::root_first_ca_path() const { return root_first_ca_path_; }
std::string FLContext::root_second_ca_path() const { return root_second_ca_path_; }

void FLContext::set_pki_verify(bool pki_verify) { pki_verify_ = pki_verify; }
bool FLContext::pki_verify() const { return pki_verify_; }

void FLContext::set_replay_attack_time_diff(uint64_t replay_attack_time_diff) {
  replay_attack_time_diff_ = replay_attack_time_diff;
}

uint64_t FLContext::replay_attack_time_diff() const { return replay_attack_time_diff_; }

std::string FLContext::equip_crl_path() const { return equip_crl_path_; }

void FLContext::set_equip_crl_path(const std::string &equip_crl_path) { equip_crl_path_ = equip_crl_path; }

void FLContext::set_scheduler_manage_port(uint16_t sched_port) {
  if (sched_port > kMaxPort) {
    MS_LOG(EXCEPTION) << "The port << " << sched_port << " is illegal.";
  }
  scheduler_manage_port_ = sched_port;
}

uint16_t FLContext::scheduler_manage_port() const { return scheduler_manage_port_; }

void FLContext::set_config_file_path(const std::string &path) { config_file_path_ = path; }

std::string FLContext::config_file_path() const { return config_file_path_; }

void FLContext::set_node_id(const std::string &node_id) {
  if (node_id.length() > kLength) {
    MS_LOG(EXCEPTION) << "The node id length can not exceed " << kLength;
  }
  node_id_ = node_id;
}

const std::string &FLContext::node_id() const { return node_id_; }

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

void FLContext::set_upload_compress_type(const std::string &upload_compress_type) {
  upload_compress_type_ = upload_compress_type;
}
std::string FLContext::upload_compress_type() const { return upload_compress_type_; }

void FLContext::set_upload_sparse_rate(float upload_sparse_rate) { upload_sparse_rate_ = upload_sparse_rate; }
float FLContext::upload_sparse_rate() const { return upload_sparse_rate_; }

void FLContext::set_download_compress_type(const std::string &download_compress_type) {
  download_compress_type_ = download_compress_type;
}
std::string FLContext::download_compress_type() const { return download_compress_type_; }

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

uint32_t FLContext::continuous_failure_times() { return continuous_failure_times_; }

void FLContext::set_feature_maps(std::vector<std::string> weight_fullnames, std::vector<std::vector<float>> weight_datas,
                                 std::vector<std::vector<size_t>> weight_shapes, std::vector<std::string> weight_types) {
  for (size_t i = 0; i < weight_fullnames.size(); i++) {
    std::string weight_fullname = weight_fullnames[i];
    std::vector<size_t> weight_shape = weight_shapes[i];
    std::string weight_type = weight_types[i];
    size_t weight_size = std::accumulate(weight_shape.begin(), weight_shape.end(), sizeof(float), std::multiplies<size_t>());
    std::vector<float> weight_data = weight_datas[i];

    Feature feature;
    feature.weight_shape = weight_shape;
    feature.weight_type = weight_type;
    feature.weight_size = weight_size;
    feature.weight_data = weight_data;

    feature_maps_[weight_fullname] = feature;
    MS_LOG(INFO) << "Weight fullname is : " << weight_fullname << ", weight size is " << weight_size
                 << ", weight shape is " << weight_shape << ", weight type is " << weight_type;

    MS_LOG(DEBUG) << "Weight data is:" << weight_data[0] << " " << weight_data[1] << " " << weight_data[2] << " ";
  }
}
std::map<std::string, Feature> &FLContext::feature_maps() {
  return feature_maps_;
}
}  // namespace fl
}  // namespace mindspore
