/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "common/communicator/ssl_client.h"

#include <sys/time.h>
#include <openssl/pem.h>
#include <openssl/sha.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <iomanip>
#include <sstream>

namespace mindspore {
namespace fl {
SSLClient::SSLClient() : ssl_ctx_(nullptr), check_time_thread_(nullptr), running_(false), is_ready_(false) {
  InitSSL();
}

SSLClient::~SSLClient() { CleanSSL(); }

void SSLClient::InitSSL() {
  if (!SSL_library_init()) {
    MS_LOG(EXCEPTION) << "SSL_library_init failed.";
  }
  if (!ERR_load_crypto_strings()) {
    MS_LOG(EXCEPTION) << "ERR_load_crypto_strings failed.";
  }
  if (!SSL_load_error_strings()) {
    MS_LOG(EXCEPTION) << "SSL_load_error_strings failed.";
  }
  if (!OpenSSL_add_all_algorithms()) {
    MS_LOG(EXCEPTION) << "OpenSSL_add_all_algorithms failed.";
  }
  ssl_ctx_ = SSL_CTX_new(SSLv23_client_method());
  if (!ssl_ctx_) {
    MS_LOG(EXCEPTION) << "SSL_CTX_new failed";
  }
  auto &ssl_config = FLContext::instance()->ssl_config();

  // 1.Parse the client's certificate and the ciphertext of key.
  std::string client_cert = kCertificateChain;
  std::string path = ssl_config.client_cert_path;
  if (!CommUtil::IsFileExists(path)) {
    MS_LOG(EXCEPTION) << "The file path of client_cert_path " << path << " is not exist.";
  }
  client_cert = path;

  // 2. Parse the client password.
  std::string client_password = FLContext::instance()->client_password();
  if (client_password.empty()) {
    MS_LOG(EXCEPTION) << "The client password's value is empty.";
  }
  EVP_PKEY *pkey = nullptr;
  X509 *cert = nullptr;
  STACK_OF(X509) *ca_stack = nullptr;
  MS_LOG(INFO) << "cliet cert: " << client_cert;
  BIO *bio = BIO_new_file(client_cert.c_str(), "rb");
  if (bio == nullptr) {
    MS_LOG(EXCEPTION) << "Read client cert file failed.";
  }
  PKCS12 *p12 = d2i_PKCS12_bio(bio, nullptr);
  if (p12 == nullptr) {
    MS_LOG(EXCEPTION) << "Create PKCS12 cert failed, please check whether the certificate is correct.";
  }
  BIO_free_all(bio);
  if (!PKCS12_parse(p12, client_password.c_str(), &pkey, &cert, &ca_stack)) {
    MS_LOG(EXCEPTION) << "PKCS12_parse failed.";
  }

  PKCS12_free(p12);
  if (cert == nullptr) {
    MS_LOG(EXCEPTION) << "the cert is nullptr";
  }
  if (pkey == nullptr) {
    MS_LOG(EXCEPTION) << "the key is nullptr";
  }

  // 3. load ca cert.
  std::string ca_path = ssl_config.ca_cert_path;
  if (!CommUtil::IsFileExists(ca_path)) {
    MS_LOG(WARNING) << "The file path of ca_cert_path " << ca_path << " is not exist.";
  }
  BIO *ca_bio = BIO_new_file(ca_path.c_str(), "r");
  if (ca_bio == nullptr) {
    MS_LOG(EXCEPTION) << "Read CA cert file failed.";
  }
  X509 *caCert = PEM_read_bio_X509(ca_bio, nullptr, nullptr, nullptr);
  std::string crl_path = ssl_config.crl_path;
  if (crl_path.empty()) {
    MS_LOG(INFO) << "The crl path is empty.";
  } else if (!CommUtil::checkCRLTime(crl_path)) {
    MS_LOG(EXCEPTION) << "check crl time failed";
  } else if (!CommUtil::VerifyCRL(cert, crl_path)) {
    MS_LOG(EXCEPTION) << "Verify crl failed.";
  }

  CommUtil::verifyCertPipeline(caCert, cert);
  InitSSLCtx(cert, pkey, ca_path);
  StartCheckCertTime(ssl_config.cert_expire_warning_time_in_day, cert);

  EVP_PKEY_free(pkey);
  (void)BIO_free(ca_bio);
}

void SSLClient::InitSSLCtx(const X509 *cert, const EVP_PKEY *pkey, std::string ca_path) {
  SSL_CTX_set_verify(ssl_ctx_, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, 0);
  if (!SSL_CTX_load_verify_locations(ssl_ctx_, ca_path.c_str(), nullptr)) {
    MS_LOG(EXCEPTION) << "SSL load ca location failed!";
  }

  auto &ssl_config = FLContext::instance()->ssl_config();
  std::string default_cipher_list = ssl_config.cipher_list;
  std::vector<std::string> ciphers = CommUtil::Split(default_cipher_list, kColon);
  if (!CommUtil::VerifyCipherList(ciphers)) {
    MS_LOG(EXCEPTION) << "The cipher is wrong.";
  }
  if (!SSL_CTX_set_cipher_list(ssl_ctx_, default_cipher_list.c_str())) {
    MS_LOG(EXCEPTION) << "SSL use set cipher list failed!";
  }
  if (!SSL_CTX_use_certificate(ssl_ctx_, const_cast<X509 *>(cert))) {
    MS_LOG(EXCEPTION) << "SSL use certificate chain file failed!";
  }

  if (!SSL_CTX_use_PrivateKey(ssl_ctx_, const_cast<EVP_PKEY *>(pkey))) {
    MS_LOG(EXCEPTION) << "SSL use private key file failed!";
  }

  if (!SSL_CTX_check_private_key(ssl_ctx_)) {
    MS_LOG(EXCEPTION) << "SSL check private key file failed!";
  }

  if (!SSL_CTX_set_options(ssl_ctx_, SSL_OP_SINGLE_DH_USE | SSL_OP_SINGLE_ECDH_USE | SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 |
                                       SSL_OP_NO_TLSv1 | SSL_OP_NO_TLSv1_1)) {
    MS_LOG(EXCEPTION) << "SSL_CTX_set_options failed.";
  }

  if (!SSL_CTX_set_mode(ssl_ctx_, SSL_MODE_AUTO_RETRY)) {
    MS_LOG(EXCEPTION) << "SSL set mode auto retry failed!";
  }

  SSL_CTX_set_security_level(ssl_ctx_, kSecurityLevel);
}

void SSLClient::CleanSSL() {
  if (ssl_ctx_ != nullptr) {
    SSL_CTX_free(ssl_ctx_);
  }
  ERR_free_strings();
  EVP_cleanup();
  CRYPTO_cleanup_all_ex_data();
  StopCheckCertTime();
}

void SSLClient::StartCheckCertTime(uint64_t cert_expire_warning_time_in_day, const X509 *cert) {
  MS_EXCEPTION_IF_NULL(cert);
  MS_LOG(INFO) << "The client start check cert.";
  int64_t interval = kCertCheckIntervalInHour;

  if (cert_expire_warning_time_in_day < kMinWarningTime || cert_expire_warning_time_in_day > kMaxWarningTime) {
    MS_LOG(EXCEPTION) << "The Certificate expiration warning time should be [7, 180]";
  }
  int64_t warning_time = static_cast<int64_t>(cert_expire_warning_time_in_day);
  MS_LOG(INFO) << "The interval time is:" << interval << ", the warning time is:" << warning_time;
  running_ = true;
  check_time_thread_ = std::make_unique<std::thread>([&, cert, interval, warning_time]() {
    while (running_) {
      if (!CommUtil::VerifyCertTime(cert, warning_time)) {
        MS_LOG(WARNING) << "Verify cert time failed.";
      }
      std::unique_lock<std::mutex> lock(mutex_);
      bool res = cond_.wait_for(lock, std::chrono::hours(interval), [&] {
        bool result = is_ready_.load();
        return result;
      });
      MS_LOG(INFO) << "Wait for res:" << res;
    }
  });
  MS_EXCEPTION_IF_NULL(check_time_thread_);
}

void SSLClient::StopCheckCertTime() {
  running_ = false;
  is_ready_ = true;
  cond_.notify_all();
  if (check_time_thread_ != nullptr) {
    check_time_thread_->join();
  }
}

SSL_CTX *SSLClient::GetSSLCtx() const { return ssl_ctx_; }
}  // namespace fl
}  // namespace mindspore
