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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMM_UTIL_H_
#define MINDSPORE_CCSRC_PS_CORE_COMM_UTIL_H_

#include <unistd.h>
#ifdef _MSC_VER
#include <iphlpapi.h>
#include <tchar.h>
#include <windows.h>
#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#endif

#include <assert.h>
#include <event2/buffer.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/keyvalq_struct.h>
#include <event2/listener.h>
#include <event2/util.h>
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pkcs12.h>
#include <openssl/rand.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/protos/comm.pb.h"
#include "common/core/cluster_config.h"
#include "common/utils/log_adapter.h"
#include "common/fl_context.h"
#include "common/utils/convert_utils_base.h"
#include "common/core/configuration.h"

namespace mindspore {
namespace fl {
struct Time {
  uint64_t time_stamp = 0;
  std::string time_str_second;
  std::string time_str_mill;
  std::string time_str_day;
};

struct FileConfig {
  uint32_t storage_type;
  std::string storage_file_path;
};

class MS_EXPORT CommUtil {
 public:
  static bool CheckIpWithRegex(const std::string &ip);
  static bool CheckIp(const std::string &ip);
  static bool CheckPort(const uint16_t &port);
  static bool SplitIpAddress(const std::string &server_address, std::string *ip, uint32_t *port);
  static void GetAvailableInterfaceAndIP(std::string *interface, std::string *ip);
  static std::string GetLoopBackInterfaceName();
  static std::string GenerateUUID();
  static std::string NodeRoleToString(const NodeRole &role);
  static NodeRole StringToNodeRole(const std::string &roleStr);
  static std::string BoolToString(bool alive);
  static bool StringToBool(const std::string &alive);
  // Check if the file exists.
  static bool IsFileExists(const std::string &file);
  // Check whether the file is empty or not.
  static bool IsFileEmpty(const std::string &file);
  // Parse the configuration file according to the key.
  static std::string ParseConfig(const Configuration &config, const std::string &key);

  // verify valid of certificate time
  static bool VerifyCertTime(const X509 *cert, int64_t time = 0);
  static bool verifyCertTimeStamp(const X509 *cert);
  // verify valid of equip certificate with CRL
  static bool VerifyCRL(const X509 *cert, const std::string &crl_path);
  static bool VerifyCommonName(const X509 *caCert, const X509 *subCert);
  static std::vector<std::string> Split(const std::string &s, char delim);
  static bool VerifyCipherList(const std::vector<std::string> &list);
  static bool verifyCertKeyID(const X509 *caCert, const X509 *subCert);
  static bool verifySingature(const X509 *caCert, const X509 *subCert);
  static bool verifyExtendedAttributes(const X509 *caCert);
  static void verifyCertPipeline(const X509 *caCert, const X509 *subCert);
  static bool checkCRLTime(const std::string &crlPath);
  static bool CreateDirectory(const std::string &directoryPath);
  static bool CheckHttpUrl(const std::string &http_url);
  static bool IsFileReadable(const std::string &file);
  template <typename T>
  static T JsonGetKeyWithException(const nlohmann::json &json, const std::string &key) {
    if (!json.contains(key)) {
      MS_LOG(EXCEPTION) << "The key " << key << "does not exist in json " << json.dump();
    }
    return json[key].get<T>();
  }
  static Time GetNowTime();
  static bool ParseAndCheckConfigJson(Configuration *file_configuration, const std::string &key,
                                      FileConfig *file_config);

 private:
  static std::random_device rd;
  static std::mt19937_64 gen;
  static std::uniform_int_distribution<> dis;
  static std::uniform_int_distribution<> dis2;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMM_UTIL_H_
