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
#include <unordered_map>
#include "vertical/vfl_context.h"
#include "common/utils/log_adapter.h"

namespace mindspore {
namespace fl {
std::shared_ptr<VFLContext> VFLContext::instance() {
  static std::shared_ptr<VFLContext> instance = nullptr;
  if (instance == nullptr) {
    instance.reset(new VFLContext());
  }
  return instance;
}

void VFLContext::LoadYamlConfig(const std::unordered_map<std::string, yaml::YamlConfigItem> &yaml_configs,
                                const std::string &yaml_config_file, const std::string &role) {
  yaml::YamlConfig config;
  config.Load(yaml_configs, yaml_config_file, role);
}

void VFLContext::set_http_server_address(const std::string &http_server_address) {
  http_server_address_ = http_server_address;
  MS_LOG(INFO) << "Local http server address is:" << http_server_address_;
}

std::string VFLContext::http_server_address() const { return http_server_address_; }

void VFLContext::set_http_server_name(const std::string &http_server_name) {
  http_server_name_ = http_server_name;
  MS_LOG(INFO) << "Local http server name is:" << http_server_name_;
}

std::string VFLContext::http_server_name() const { return http_server_name_; }

void VFLContext::set_ssl_config(const SslConfig &config) { ssl_config_ = config; }

const SslConfig &VFLContext::ssl_config() const { return ssl_config_; }

bool VFLContext::enable_ssl() const { return enable_ssl_; }

void VFLContext::set_enable_ssl(bool enabled) { enable_ssl_ = enabled; }

std::string VFLContext::client_password() const { return client_password_; }
void VFLContext::set_client_password(const std::string &password) { client_password_ = password; }

std::string VFLContext::server_password() const { return server_password_; }
void VFLContext::set_server_password(const std::string &password) { server_password_ = password; }

std::string VFLContext::http_url_prefix() const { return http_url_prefix_; }

void VFLContext::set_http_url_prefix(const std::string &http_url_prefix) { http_url_prefix_ = http_url_prefix; }

void VFLContext::set_remote_server_address(const std::map<std::string, std::string> &remote_server_address) {
  remote_server_address_ = remote_server_address;
  MS_LOG(INFO) << "Remote http server address is:" << remote_server_address_;
}

std::map<std::string, std::string> VFLContext::remote_server_address() const { return remote_server_address_; }
}  // namespace fl
}  // namespace mindspore
