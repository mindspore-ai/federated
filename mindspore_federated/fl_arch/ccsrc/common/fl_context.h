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

#ifndef MINDSPORE_CCSRC_FL_CONTEXT_H_
#define MINDSPORE_CCSRC_FL_CONTEXT_H_

#include <map>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "common/constants.h"
#include "common/common.h"
#include "common/core/cluster_config.h"
#include "common/core/yaml_config.h"
#include "distributed_cache/distributed_cache.h"

namespace mindspore {
namespace fl {
using DistributedCacheConfig = cache::DistributedCacheConfig;

struct FeatureInfo {
  std::vector<size_t> weight_shape;
  std::string weight_type;
  size_t weight_size;
};

struct AggregationConfig {
  std::string aggregation_type = kFedAvgAggregation;
  float iid_rate = 0.0f;
  size_t total_client_num = 1;
};

struct EncryptConfig {
  std::string encrypt_type = kNotEncryptType;
  // pw encrypt
  float share_secrets_ratio = 1.0f;
  uint64_t cipher_time_window = 1;
  uint64_t reconstruct_secrets_threshold = 1;
  // dp encrypt
  float dp_eps = 50.0f;
  float dp_delta = 0.01f;
  float dp_norm_clip = 1.0f;
  // sign ds
  float sign_k = 0.01f;
  float sign_eps = 100.0f;
  float sign_thr_ratio = 0.6f;
  float sign_global_lr = 0.1f;
  uint64_t sign_dim_out = 0;
};

struct CompressionConfig {
  std::string upload_compress_type = kNoCompressType;
  float upload_sparse_rate = 0.4f;
  std::string download_compress_type = kNoCompressType;
};

struct SslConfig {
  // Server certificate file path.
  std::string server_cert_path;
  // Client certificate file path.
  std::string client_cert_path;
  // CA Server certificate file path.
  std::string ca_cert_path;
  // CRL certificate file path.
  std::string crl_path;
  // Encryption suite supported by ssl.
  std::string cipher_list;
  // Warning time before the certificate expires.
  uint64_t cert_expire_warning_time_in_day = kCertExpireWarningTimeInDay;
};

struct ClientVerifyConfig {
  bool pki_verify = false;
  std::string root_first_ca_path;
  std::string root_second_ca_path;
  std::string equip_crl_path;
  uint64_t replay_attack_time_diff = 600000;
};

struct UnsupervisedConfig {
  // unsupervised cluster client number of federeated learning. Default:1000.
  uint64_t cluster_client_num = 0;
  // unsupervised eval type of federeated learning. Default:silhouette_score.
  std::string eval_type = kNotEvalType;
};

constexpr char kEnvRole[] = "MS_ROLE";
constexpr char kEnvRoleOfServer[] = "MS_SERVER";
constexpr char kEnvRoleOfWorker[] = "MS_WORKER";
constexpr char kEnvRoleOfScheduler[] = "MS_SCHED";

// Use binary data to represent federated learning server's context so that we can judge which round resets the
// iteration. From right to left, each bit stands for:
// 1: Server is in federated learning mode.
// 2: Server is in mixed training mode.
// 3: Server enables pairwise encrypt algorithm.
// For example: 1010 stands for that the server is in federated learning mode and pairwise encrypt algorithm is enabled.
enum class ResetterRound { kNoNeedToReset, kUpdateModel, kReconstructSeccrets, kPushWeight, kPushMetrics };
const std::map<uint32_t, ResetterRound> kServerContextToResetRoundMap = {{0b0010, ResetterRound::kUpdateModel},
                                                                         {0b1010, ResetterRound::kReconstructSeccrets},
                                                                         {0b1100, ResetterRound::kPushMetrics},
                                                                         {0b0100, ResetterRound::kPushMetrics}};

class MS_EXPORT FLContext {
 public:
  ~FLContext() = default;
  FLContext(FLContext const &) = delete;
  FLContext &operator=(const FLContext &) = delete;
  static std::shared_ptr<FLContext> instance();

  void LoadYamlConfig(const std::unordered_map<std::string, yaml::YamlConfigItem> &yaml_configs,
                      const std::string &yaml_config_file, const std::string &role);

  void Reset();
  std::string ms_role() const;
  bool is_worker() const;
  bool is_server() const;
  bool is_scheduler() const;

  // In new server framework, process role, worker number, server number, scheduler ip and scheduler port should be set
  // by fl_context.
  void set_server_mode(const std::string &server_mode);
  const std::string &server_mode() const;

  const std::string &encrypt_type() const;

  void set_ms_role(const std::string &role);

  void set_tcp_server_ip(const std::string &tcp_server_ip);
  std::string tcp_server_ip() const;

  // Generate which round should reset the iteration.
  void GenerateResetterRound();
  ResetterRound resetter_round() const;

  void set_http_server_address(const std::string &http_server_address);
  std::string http_server_address() const;

  void set_start_fl_job_threshold(uint64_t start_fl_job_threshold);
  uint64_t start_fl_job_threshold() const;

  void set_start_fl_job_time_window(uint64_t start_fl_job_time_window);
  uint64_t start_fl_job_time_window() const;

  void set_update_model_ratio(float update_model_ratio);
  float update_model_ratio() const;

  void set_update_model_time_window(uint64_t update_model_time_window);
  uint64_t update_model_time_window() const;

  void set_fl_name(const std::string &fl_name);
  const std::string &fl_name() const;

  void set_aggregation_config(const AggregationConfig &config);
  const AggregationConfig &aggregation_config() const;
  const std::string &aggregation_type() const;
  const float &iid_rate() const;
  const size_t &total_client_num() const;

  // Set the iteration number of the federated learning.
  void set_fl_iteration_num(uint64_t fl_iteration_num);
  uint64_t fl_iteration_num() const;

  // Set the training epoch number of the client.
  void set_client_epoch_num(uint64_t client_epoch_num);
  uint64_t client_epoch_num() const;

  void set_client_verify_config(const ClientVerifyConfig &config);
  const ClientVerifyConfig &client_verify_config() const;

  void set_distributed_cache_config(const DistributedCacheConfig &config);
  const DistributedCacheConfig &distributed_cache_config() const;

  void set_encrypt_config(const EncryptConfig &config);
  const EncryptConfig &encrypt_config() const;

  void set_compression_config(const CompressionConfig &config);
  const CompressionConfig &compression_config() const;

  // Set the data batch size of the client.
  void set_client_batch_size(uint64_t client_batch_size);
  uint64_t client_batch_size() const;

  void set_client_learning_rate(float client_learning_rate);
  float client_learning_rate() const;

  void set_max_connection_num(uint64_t max_connection_num);
  uint64_t max_connection_num() const;

  // Set true if using secure aggregation for federated learning.
  void set_secure_aggregation(bool secure_aggregation);
  bool secure_aggregation() const;

  ClusterConfig &cluster_config();

  bool pki_verify() const;

  void set_scheduler_manage_address(const std::string &manage_address);
  std::string scheduler_manage_address() const;

  bool enable_ssl() const;
  void set_enable_ssl(bool enabled);

  void set_ssl_config(const SslConfig &config);
  const SslConfig &ssl_config() const;

  std::string client_password() const;
  void set_client_password(const std::string &password);

  std::string server_password() const;
  void set_server_password(const std::string &password);

  std::string http_url_prefix() const;
  void set_http_url_prefix(const std::string &http_url_prefix);

  void set_global_iteration_time_window(const uint64_t &global_iteration_time_window);
  uint64_t global_iteration_time_window() const;

  std::string checkpoint_dir() const;
  void set_checkpoint_dir(const std::string &checkpoint_dir);

  void set_instance_name(const std::string &instance_name);
  const std::string &instance_name() const;

  void set_participation_time_level(const std::string &participation_time_level);
  const std::string &participation_time_level();

  void set_continuous_failure_times(uint32_t continuous_failure_times);
  uint32_t continuous_failure_times() const;

  void set_metrics_file(const std::string &metrics_file);
  const std::string &metrics_file();

  void set_failure_event_file(const std::string &failure_event_file);
  const std::string &failure_event_file();

  void set_data_rate_dir(const std::string &data_rate_dir);
  const std::string &data_rate_dir();

  void set_unsupervised_config(const UnsupervisedConfig &unsupervised_config);
  UnsupervisedConfig unsupervised_config() const;

 private:
  FLContext() = default;

  // The server process's role.
  std::string role_ = kEnvRoleOfServer;

  // Server mode which could be Federated Learning and Hybrid Training mode.
  std::string server_mode_;

  // The round which will reset the iteration. Used in federated learning for now.
  ResetterRound resetter_round_ = ResetterRound::kNoNeedToReset;

  // Http port of federated learning server.
  std::string http_server_address_;

  std::string tcp_server_ip_ = "127.0.0.1";

  // Whether this process is the federated client. Used in cross-silo scenario of federated learning.
  bool fl_client_enable_ = false;

  // Federated learning job name.
  std::string fl_name_;

  // The threshold count of startFLJob round. Used in federated learning for now.
  uint64_t start_fl_job_threshold_ = 0;

  // The time window of startFLJob round in millisecond.
  uint64_t start_fl_job_time_window_ = 300000;

  // Update model threshold is a certain ratio of start_fl_job threshold which is set as update_model_ratio_.
  float update_model_ratio_ = 1.0;

  // The time window of updateModel round in millisecond.
  uint64_t update_model_time_window_ = 300000;

  // Share model threshold is a certain ratio of share secrets threshold which is set as share_secrets_ratio_.
  float share_secrets_ratio_ = 1.0;

  // The time window of each cipher round in millisecond.
  uint64_t cipher_time_window_ = 300000;

  // The threshold count of reconstruct secrets round. Used in federated learning for now.
  uint64_t reconstruct_secrets_threshold_ = 2000;

  // Iteration number of federeated learning, which is the number of interactions between client and server.
  uint64_t fl_iteration_num_ = 20;

  // Client training epoch number. Used in federated learning for now.
  uint64_t client_epoch_num_ = 25;

  // Client training data batch size. Used in federated learning for now.
  uint64_t client_batch_size_ = 32;

  // Client training learning rate. Used in federated learning for now.
  float client_learning_rate_ = 0.001;

  uint64_t max_collection_num_ = kConnectionNumDefault;

  // Whether to use secure aggregation algorithm. Used in federated learning for now.
  bool secure_aggregation_ = false;

  // The cluster config read through environment variables, the value does not change.
  std::unique_ptr<ClusterConfig> cluster_config_ = nullptr;

  // The port used by scheduler to receive http requests for scale out or scale in.
  std::string scheduler_manage_address_;

  // The path of the configuration file, used to configure the certification path and persistent storage type, etc.
  std::string config_file_path_;

  // Unique id of the node
  std::string node_id_;

  // Whether to enable ssl for network communication.
  bool enable_ssl_ = false;
  // Password used to decode p12 file.
  std::string client_password_;
  // Password used to decode p12 file.
  std::string server_password_;
  // http url prefix for http communication
  std::string http_url_prefix_;

  // The time window of startFLJob round in millisecond.
  uint64_t global_iteration_time_window_ = 3600000;

  // Hyper parameters for upload compression.
  std::string upload_compress_type_ = kNoCompressType;
  float upload_sparse_rate_ = 0.4f;
  // Hyper parameters for download compression.
  std::string download_compress_type_ = kNoCompressType;

  // directory of server checkpoint
  std::string checkpoint_dir_;

  // The name of instance
  std::string instance_name_;

  // The participation time level
  std::string participation_time_level_ = "5,15";

  // The times of iteration continuous failure
  uint32_t continuous_failure_times_ = 10;

  DistributedCacheConfig distributed_cache_config_;
  SslConfig ssl_config_;
  // server config
  EncryptConfig encrypt_config_;
  CompressionConfig compression_config_;
  ClientVerifyConfig client_verify_config_;
  AggregationConfig aggregation_config_;
  UnsupervisedConfig unsupervised_config_;

  std::string metrics_file_ = "metrics.json";
  std::string failure_event_file_ = "event.txt";
  std::string data_rate_dir_ = "";
  void CheckDPEncrypt(const EncryptConfig &config) const;
  void CheckSignDsEncrypt(const EncryptConfig &config) const;
  void CheckPWEncrypt(const EncryptConfig &config) const;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_CONTEXT_H_
