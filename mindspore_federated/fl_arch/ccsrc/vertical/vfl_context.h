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

#ifndef MINDSPORE_FL_ARCH_CCSRC_VERTICAL_VFL_CONTEXT_H_
#define MINDSPORE_FL_ARCH_CCSRC_VERTICAL_VFL_CONTEXT_H_

#include <map>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "common/constants.h"
#include "common/common.h"
#include "common/core/yaml_config.h"

namespace mindspore {
namespace fl {
constexpr char kEnvRoleOfLeaderTrainer[] = "MS_LEADER_TRAINER";
constexpr char kEnvRoleOfFollowerTrainer[] = "MS_FOLLOWER_TRAINER";

class MS_EXPORT VFLContext {
 public:
  ~VFLContext() = default;
  VFLContext(VFLContext const &) = delete;
  VFLContext &operator=(const VFLContext &) = delete;
  static std::shared_ptr<VFLContext> instance();

  void LoadYamlConfig(const std::unordered_map<std::string, yaml::YamlConfigItem> &yaml_configs,
                      const std::string &yaml_config_file, const std::string &role);

  void set_http_server_address(const std::string &http_server_address);
  std::string http_server_address() const;

  void set_http_server_name(const std::string &http_server_name);
  std::string http_server_name() const;

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

  void set_remote_server_address(const std::map<std::string, std::string> &remote_server_address);
  std::map<std::string, std::string> remote_server_address() const;

 private:
  VFLContext() = default;

  // The server process's role.
  std::string role_ = kEnvRoleOfLeaderTrainer;

  // Http port of federated learning server.
  std::string http_server_address_;
  std::string http_server_name_;

  // Whether to enable ssl for network communication.
  bool enable_ssl_ = false;
  // Password used to decode p12 file.
  std::string client_password_;
  // Password used to decode p12 file.
  std::string server_password_;
  // http url prefix for http communication
  std::string http_url_prefix_;
  SslConfig ssl_config_;
  std::map<std::string, std::string> remote_server_address_;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_VFL_CONTEXT_H_
