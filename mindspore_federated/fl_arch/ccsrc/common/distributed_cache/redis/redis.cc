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

#include "distributed_cache/redis/redis.h"
#include <algorithm>
#include <utility>
#include "common/utils/log_adapter.h"
#include "common/core/comm_util.h"
#include "common/common.h"
#include "common/exit_handler.h"

namespace mindspore {
namespace fl {
namespace cache {
RedisReply::RedisReply(redisReply *reply) noexcept {
  if (reply != nullptr) {
    redis_reply_ = std::unique_ptr<redisReply, RedisReply::RedisReplyDeleter>(reply);
  }
}

RedisReply::RedisReply(RedisReply &&other) noexcept { redis_reply_.swap(other.redis_reply_); }

RedisReply &RedisReply::operator=(RedisReply &&other) noexcept {
  redis_reply_.swap(other.redis_reply_);
  return *this;
}

int RedisReply::GetType() const {
  if (redis_reply_ == nullptr) {
    return REDIS_REPLY_ERROR;
  }
  return redis_reply_->type;
}

bool RedisReply::IsValid() const { return redis_reply_ && redis_reply_->type != REDIS_REPLY_ERROR; }

bool RedisReply::GetStatus(std::string *value) const {
  if (value == nullptr) {
    return false;
  }
  if (redis_reply_ == nullptr) {
    return false;
  }
  if (redis_reply_->type != REDIS_REPLY_STATUS) {
    MS_LOG(ERROR) << "Get status value failed, reply type " << redis_reply_->type << " is not status "
                  << REDIS_REPLY_STATUS;
    return false;
  }
  *value = std::string(redis_reply_->str, redis_reply_->len);
  return true;
}

std::string RedisReply::GetError() const {
  if (redis_reply_ == nullptr) {
    return "Construct reply failed";
  }
  if (redis_reply_->type != REDIS_REPLY_ERROR) {
    MS_LOG(ERROR) << "reply is has no error";
    return "reply is has no error";
  }
  return std::string(redis_reply_->str, redis_reply_->len);
}

bool RedisReply::GetInteger(uint64_t *value) const {
  if (value == nullptr) {
    return false;
  }
  if (redis_reply_ == nullptr) {
    return false;
  }
  if (redis_reply_->type != REDIS_REPLY_INTEGER) {
    MS_LOG(ERROR) << "Get integer value failed, reply type " << redis_reply_->type << " is not integer "
                  << REDIS_REPLY_INTEGER;
    return false;
  }
  *value = static_cast<uint64_t>(redis_reply_->integer);
  return true;
}

bool RedisReply::GetString(std::string *value) const {
  if (value == nullptr) {
    return false;
  }
  if (redis_reply_ == nullptr) {
    return false;
  }
  if (redis_reply_->type != REDIS_REPLY_STRING) {
    MS_LOG(ERROR) << "Get string value failed, reply type " << redis_reply_->type << " is not string "
                  << REDIS_REPLY_STRING;
    return false;
  }
  *value = std::string(redis_reply_->str, redis_reply_->len);
  return true;
}

bool RedisReply::GetArray(std::vector<std::string> *value) const {
  if (value == nullptr) {
    return false;
  }
  if (redis_reply_ == nullptr) {
    return false;
  }
  if (redis_reply_->type != REDIS_REPLY_ARRAY) {
    MS_LOG(ERROR) << "Get array value failed, reply type " << redis_reply_->type << " is not array "
                  << REDIS_REPLY_ARRAY;
    return false;
  }
  value->clear();
  for (size_t i = 0; i < redis_reply_->elements; i++) {
    auto member = redis_reply_->element[i];
    if (member == nullptr) {
      MS_LOG(ERROR) << "Get array value failed, element cannot be nullptr";
      return false;
    }
    if (member->type != REDIS_REPLY_STRING) {
      MS_LOG(ERROR) << "Get array value failed, elements type should be string " << REDIS_REPLY_STRING
                    << ", member type: " << member->type;
      return false;
    }
    value->push_back(std::string(member->str, member->len));
  }
  return true;
}

bool RedisReply::GetMap(std::unordered_map<std::string, std::string> *value) const {
  if (value == nullptr) {
    return false;
  }
  if (redis_reply_ == nullptr) {
    return false;
  }
  if (redis_reply_->type != REDIS_REPLY_ARRAY) {
    MS_LOG(ERROR) << "Get map value failed, reply type " << redis_reply_->type << " is not array " << REDIS_REPLY_ARRAY;
    return false;
  }
  if (redis_reply_->elements % 2 != 0) {
    MS_LOG(ERROR) << "Get map value failed, element size " << redis_reply_->elements << " is not even";
    return false;
  }
  value->clear();
  for (size_t i = 0; i + 1 < redis_reply_->elements; i += 2) {
    auto key = redis_reply_->element[i];
    auto val = redis_reply_->element[i + 1];
    if (key == nullptr || val == nullptr) {
      MS_LOG(ERROR) << "Get map value failed, element cannot be nullptr";
      return false;
    }
    if (key->type != REDIS_REPLY_STRING || val->type != REDIS_REPLY_STRING) {
      MS_LOG(ERROR) << "Get map value failed, elements type should be string " << REDIS_REPLY_STRING
                    << ", key type: " << key->type << ", value type: " << val->type;
      return false;
    }
    value->emplace(std::make_pair(std::string(key->str, key->len), std::string(val->str, val->len)));
  }
  return true;
}

bool RedisReply::IsNil() const { return redis_reply_ != nullptr && redis_reply_->type == REDIS_REPLY_NIL; }

RedisClient::RedisClient(const std::string &server_address, redisSSLContext *ssl_context, int64_t timeout)
    : server_address_(server_address), ssl_context_(ssl_context), timeout_(timeout) {}

RedisClient::~RedisClient() { Disconnect(); }

CacheStatus RedisClient::Connect(bool retry_connect) {
  bool is_unix_address = IsUnixAddress(server_address_);
  std::string ip;
  uint32_t port;
  if (!is_unix_address) {
    if (!CommUtil::SplitIpAddress(server_address_, &ip, &port)) {
      auto reason = "Connection error: invalid redis server address: " + server_address_;
      MS_LOG(ERROR) << reason;
      return {kCacheNetErr, reason};
    }
  }
  int64_t retry_times = timeout_;
  if (retry_times <= 0 || !retry_connect) {
    retry_times = 1;
  }
  const std::string refused_str = "Connection refused";
  for (int64_t i = 0; i < retry_times; i++) {
    if (ExitHandler::Instance().HasStopped()) {
      auto reason =
        std::string("Connection canceled: ") + "receive signal " + std::to_string(ExitHandler::Instance().GetSignal());
      MS_LOG(ERROR) << reason;
      return {kCacheNetErr, reason};
    }
    if (redis_context_ != nullptr) {
      redisFree(redis_context_);
      redis_context_ = nullptr;
    }
    if (is_unix_address) {
      redis_context_ = redisConnectUnix(server_address_.c_str());
    } else {
      redis_context_ = redisConnect(ip.c_str(), static_cast<int>(port));
    }
    if (redis_context_ == nullptr) {
      auto reason = "Connection error: cannot allocate redis context, redis address: " + server_address_;
      MS_LOG(ERROR) << reason;
      return {kCacheNetErr, reason};
    }
    if (!redis_context_->err) {
      break;
    }
    if (redis_context_->errstr != refused_str) {
      auto reason = std::string("Connection error: ") + redis_context_->errstr + ", redis address: " + server_address_;
      MS_LOG(ERROR) << reason;
      return {kCacheNetErr, reason};
    }
    if (i < retry_times - 1) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  if (redis_context_->err) {
    auto reason = std::string("Connection error: ") + redis_context_->errstr + ", redis address: " + server_address_;
    MS_LOG(ERROR) << reason;
    return {kCacheNetErr, reason};
  }
  if (ssl_context_ != nullptr) {
    if (redisInitiateSSLWithContext(redis_context_, ssl_context_) != REDIS_OK) {
      auto reason =
        std::string("Initialize SSL error: ") + redis_context_->errstr + ", redis address: " + server_address_;
      MS_LOG(ERROR) << reason;
      return {kCacheNetErr, reason};
    }
  }
  if (redisEnableKeepAlive(redis_context_) != REDIS_OK) {
    auto reason = "Connection error: failed to enable keep alive option";
    MS_LOG(ERROR) << reason;
    return {kCacheNetErr, reason};
  }
  return kCacheSuccess;
}

bool RedisClient::IsUnixAddress(const std::string &server_address) {
  const std::string unix_prefix = "unix:";
  if (server_address.find(unix_prefix) == 0 && server_address > unix_prefix) {
    return true;
  }
  return false;
}

bool RedisClient::IsValid() { return redis_context_ != nullptr && !redis_context_->err; }

CacheStatus RedisClient::Reconnect() {
  std::unique_lock<std::mutex> lock(lock_);
  return ReconnectInner();
}

CacheStatus RedisClient::ReconnectInner() {
  if (redis_context_ == nullptr) {
    return Connect(false);
  }
  if (redisReconnect(redis_context_) != REDIS_OK) {
    std::string reason = redis_context_->errstr;
    return {kCacheNetErr, reason};
  }
  if (ssl_context_ != nullptr) {
    if (redisInitiateSSLWithContext(redis_context_, ssl_context_) != REDIS_OK) {
      auto reason =
        std::string("Initialize SSL error: ") + redis_context_->errstr + ", redis address: " + server_address_;
      MS_LOG(ERROR) << reason;
      return {kCacheNetErr, reason};
    }
  }
  if (redisEnableKeepAlive(redis_context_) != REDIS_OK) {
    auto reason = "Connection error: failed to enable keep alive option";
    MS_LOG(ERROR) << reason;
    return {kCacheNetErr, reason};
  }
  return kCacheSuccess;
}

void RedisClient::Disconnect() {
  if (!redis_context_) {
    redisFree(redis_context_);
    redis_context_ = nullptr;
  }
}

RedisReply RedisClient::RunCommand(int argc, const char **argv, const size_t *argvlen) {
  MS_EXCEPTION_IF_NULL(argv);
  MS_EXCEPTION_IF_NULL(argvlen);
  std::unique_lock<std::mutex> lock(lock_);
  if (!IsValid()) {
    auto status = ReconnectInner();
    if (!status.IsSuccess()) {
      MS_LOG(ERROR) << "Init redis command failed, failed to reconnect to redis server: " << status.GetDetail();
      return {};
    }
  }
  auto reply = reinterpret_cast<redisReply *>(redisCommandArgv(redis_context_, argc, argv, argvlen));
  auto redis_reply = RedisReply(reply);
  if (!IsValid()) {
    auto status = ReconnectInner();
    if (!status.IsSuccess()) {
      MS_LOG(ERROR) << "Init redis command failed, failed to reconnect to redis server: " << status.GetDetail();
      return {};
    }
    reply = reinterpret_cast<redisReply *>(redisCommandArgv(redis_context_, argc, argv, argvlen));
    redis_reply = RedisReply(reply);
  }
  return redis_reply;
}

RedisReply RedisClient::RunCommand(const std::vector<std::string> &args) {
  std::vector<const char *> argv;
  std::transform(args.begin(), args.end(), std::back_inserter(argv),
                 [](const std::string &item) { return item.c_str(); });

  std::vector<size_t> argvlen;
  std::transform(args.begin(), args.end(), std::back_inserter(argvlen),
                 [](const std::string &item) { return item.size(); });
  return RunCommand(static_cast<int>(argv.size()), argv.data(), argvlen.data());
}

RedisReply RedisClient::Eval(const std::string &script, const std::vector<std::string> &keys,
                             const std::vector<std::string> &args) {
  std::vector<const char *> argv;
  argv.push_back(script.c_str());
  std::transform(keys.begin(), keys.end(), std::back_inserter(argv),
                 [](const std::string &item) { return item.c_str(); });
  std::transform(args.begin(), args.end(), std::back_inserter(argv),
                 [](const std::string &item) { return item.c_str(); });

  std::vector<size_t> argvlen;
  argvlen.push_back(script.size());
  std::transform(keys.begin(), keys.end(), std::back_inserter(argvlen),
                 [](const std::string &item) { return item.size(); });
  std::transform(args.begin(), args.end(), std::back_inserter(argvlen),
                 [](const std::string &item) { return item.size(); });
  return RunCommand(static_cast<int>(argv.size()), argv.data(), argvlen.data());
}

CacheStatus RedisClient::Del(const std::vector<std::string> &keys) {
  std::vector<std::string> args;
  args.emplace_back("DEL");
  std::copy(keys.begin(), keys.end(), std::back_inserter(args));
  RedisReply reply = RunCommand(args);
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::Expire(const std::string &key, uint64_t seconds) {
  RedisReply reply = RunCommand({"EXPIRE", key, std::to_string(seconds)});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::SAdd(const std::string &key, const std::string &member) {
  RedisReply reply = RunCommand({"SADD", key, member});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  uint64_t int_value = 0;
  if (!reply.GetInteger(&int_value)) {
    MS_LOG(WARNING) << "Failed to call SADD " << key << " " << member;
    return kCacheInnerErr;
  }
  if (int_value == 0) {
    return kCacheExist;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::SIsMember(const std::string &key, const std::string &member, bool *value) {
  RedisReply reply = RunCommand({"SISMEMBER", key, member});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  uint64_t int_value = 0;
  if (!reply.GetInteger(&int_value)) {
    MS_LOG(WARNING) << "Failed to call SISMEMBER " << key << " " << member;
    return kCacheInnerErr;
  }
  *value = (int_value != 0);
  return kCacheSuccess;
}

CacheStatus RedisClient::SMembers(const std::string &key, std::vector<std::string> *members) {
  MS_EXCEPTION_IF_NULL(members);
  RedisReply reply = RunCommand({"SMEMBERS", key});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  if (!reply.GetArray(members)) {
    MS_LOG(WARNING) << "Failed to call SMEMBERS " << key;
    return kCacheInnerErr;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::HExists(const std::string &key, const std::string &filed, bool *bool_value) {
  RedisReply reply = RunCommand({"HEXISTS", key, filed});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  uint64_t value = 0;
  if (!reply.GetInteger(&value)) {
    MS_LOG(WARNING) << "Failed to call HEXIST " << key << " " << filed;
    return kCacheInnerErr;
  }
  *bool_value = (value != 0);
  return kCacheSuccess;
}

CacheStatus RedisClient::HSet(const std::string &key, const std::string &filed, const std::string &value) {
  RedisReply reply = RunCommand({"HSET", key, filed, value});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::HSetNx(const std::string &key, const std::string &filed, const std::string &value) {
  RedisReply reply = RunCommand({"HSETNX", key, filed, value});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  uint64_t ret_value = 0;
  if (!reply.GetInteger(&ret_value)) {
    MS_LOG(WARNING) << "Failed to call HSETNX " << key << " " << filed;
    return kCacheInnerErr;
  }
  if (ret_value == 0) {
    return kCacheExist;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::HMSet(const std::string &key, const std::unordered_map<std::string, std::string> &items) {
  std::vector<std::string> args = {"HMSET", key};
  for (auto &item : items) {
    args.push_back(item.first);
    args.push_back(item.second);
  }
  RedisReply reply = RunCommand(args);
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::HGet(const std::string &key, const std::string &filed, std::string *value) {
  MS_EXCEPTION_IF_NULL(value);
  RedisReply reply = RunCommand({"HGET", key, filed});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  if (reply.IsNil()) {
    return kCacheNil;
  }
  if (!reply.GetString(value)) {
    MS_LOG(WARNING) << "Failed to call HGet " << key << " " << filed;
    return kCacheInnerErr;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::HGetAll(const std::string &key, std::unordered_map<std::string, std::string> *items) {
  MS_EXCEPTION_IF_NULL(items);
  RedisReply reply = RunCommand({"HGETALL", key});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  if (!reply.GetMap(items)) {
    MS_LOG(WARNING) << "Failed to call HGETALL " << key;
    return kCacheInnerErr;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::HIncr(const std::string &key, const std::string &filed, uint64_t *new_value) {
  RedisReply reply = RunCommand({"HINCRBY", key, filed, "1"});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  uint64_t ret_value = 0;
  if (!reply.GetInteger(&ret_value)) {
    MS_LOG(WARNING) << "Failed to call HINCRBY " << key;
    return kCacheInnerErr;
  }
  *new_value = static_cast<uint64_t>(ret_value);
  return kCacheSuccess;
}

CacheStatus RedisClient::HDel(const std::string &key, const std::string &filed) {
  RedisReply reply = RunCommand({"HDEL", key, filed});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::Get(const std::string &key, std::string *value) {
  RedisReply reply = RunCommand({"GET", key});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  if (reply.IsNil()) {
    return kCacheNil;
  }
  if (!reply.GetString(value)) {
    MS_LOG(WARNING) << "Failed to call GET " << key;
    return kCacheInnerErr;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::SetEx(const std::string &key, const std::string &value, uint64_t seconds) {
  RedisReply reply = RunCommand({"SET", key, value, "EX", std::to_string(seconds)});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::SetNx(const std::string &key, const std::string &value) {
  RedisReply reply = RunCommand({"SET", key, value, "NX"});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  if (reply.IsNil()) {
    return kCacheExist;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::SetExNx(const std::string &key, const std::string &value, uint64_t seconds) {
  RedisReply reply = RunCommand({"SET", key, value, "EX", std::to_string(seconds), "NX"});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  if (reply.IsNil()) {
    return kCacheExist;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::Incr(const std::string &key, uint64_t *new_value) {
  RedisReply reply = RunCommand({"INCR", key});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  if (!reply.GetInteger(new_value)) {
    MS_LOG(WARNING) << "Failed to call INCR " << key;
    return kCacheInnerErr;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::LPush(const std::string &key, const std::string &value) {
  RedisReply reply = RunCommand({"LPUSH", key, value});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  uint64_t ret_value = 0;
  if (!reply.GetInteger(&ret_value)) {
    MS_LOG(WARNING) << "Failed to call LPush key:" << key;
    return kCacheInnerErr;
  }
  if (ret_value == 0) {
    return kCacheInnerErr;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::LRange(const std::string &key, size_t start, size_t end, std::vector<std::string> *items) {
  RedisReply reply = RunCommand({"LRANGE", key, std::to_string(start), std::to_string(end)});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  if (!reply.GetArray(items)) {
    MS_LOG(WARNING) << "Failed to call LRange " << key;
    return kCacheInnerErr;
  }
  return kCacheSuccess;
}

CacheStatus RedisClient::LTrim(const std::string &key, size_t start, size_t end) {
  RedisReply reply = RunCommand({"LTRIM", key, std::to_string(start), std::to_string(end)});
  if (!reply.IsValid()) {
    MS_LOG(WARNING) << "Reply invalid: " << reply.GetError();
    return kCacheNetErr;
  }
  return kCacheSuccess;
}

RedisDistributedCache::~RedisDistributedCache() {
  client_pool_.clear();
  if (ssl_context_ != nullptr) {
    redisFreeSSLContext(ssl_context_);
    ssl_context_ = nullptr;
  }
}

CacheStatus RedisDistributedCache::ParseSSLConfig(const std::unordered_map<std::string, std::string> &configs,
                                                  RedisSSLConfig *ssl_config_p) {
  RedisSSLConfig &ssl_config = *ssl_config_p;
  auto get_item = [&configs](const std::string &name, bool required, std::string *val) {
    auto it = configs.find(name);
    if (it != configs.end()) {
      *val = it->second;
    }
  };
  get_item("cacert_filename", false, &ssl_config.cacert_filename);
  get_item("capath", false, &ssl_config.capath);
  get_item("cert_filename", false, &ssl_config.cert_filename);
  get_item("private_key_filename", false, &ssl_config.private_key_filename);
  get_item("server_name", false, &ssl_config.server_name);
  return kCacheSuccess;
}

bool RedisDistributedCache::Init(const DistributedCacheConfig &cache_config, int64_t timeout) {
  cache_config_ = cache_config;
  if (cache_config_.address.empty()) {
    MS_LOG_ERROR << "Invalid input parameter, server_address: " << cache_config_.address;
    return false;
  }
  if (FLContext::instance()->enable_ssl()) {
    RedisSSLConfig ssl_config;
    auto ret = ParseSSLConfig(cache_config_.configs, &ssl_config);
    if (!ret.IsSuccess()) {
      MS_LOG_ERROR << "Failed to parse ssl config, detail: " << ret.GetDetail();
      return false;
    }
    redisInitOpenSSL();
    redisSSLContextError ssl_error;
    auto c_str = [](const std::string &str) -> const char * { return str.empty() ? nullptr : str.c_str(); };
    ssl_context_ = redisCreateSSLContext(c_str(ssl_config.cacert_filename), c_str(ssl_config.capath),
                                         c_str(ssl_config.cert_filename), c_str(ssl_config.private_key_filename),
                                         c_str(ssl_config.server_name), &ssl_error);
    if (!ssl_context_) {
      MS_LOG_ERROR << "Redis SSL context error: " << redisSSLContextGetError(ssl_error);
      return false;
    }
  }
  MS_LOG_INFO << "Try connect to redis sever " << cache_config_.address << ", retry time in seconds " << timeout;
  for (size_t i = 0; i < thread_pool_size_; i++) {
    auto client = std::make_shared<RedisClient>(cache_config_.address, ssl_context_, timeout);
    if (client == nullptr) {
      MS_LOG_ERROR << "Failed to create RedisClient object";
      return false;
    }
    auto ret = client->Connect(true);
    if (!ret.IsSuccess()) {
      MS_LOG_ERROR << "Connect to redis server failed, server address: " << cache_config_.address;
      return false;
    }
    client_pool_.push_back(client);
  }
  MS_LOG_INFO << "Connect to redis sever " << cache_config_.address << " successfully";
  return true;
}

std::shared_ptr<RedisClientBase> RedisDistributedCache::GetOneClient() {
  if (client_pool_.empty()) {
    return nullptr;
  }
  auto ret_index = cur_client_ret_index_++;
  ret_index = ret_index % client_pool_.size();
  return client_pool_[ret_index];
}

bool RedisDistributedCache::HasInvalid() const {
  return std::any_of(client_pool_.begin(), client_pool_.end(),
                     [](const std::shared_ptr<RedisClient> &item) { return !item->IsValid(); });
}

CacheStatus RedisDistributedCache::RetryConnect() {
  for (auto &item : client_pool_) {
    if (!item->IsValid()) {
      auto status = item->Reconnect();
      if (!status.IsSuccess()) {
        return status;
      }
    }
  }
  return kCacheSuccess;
}

void RedisDistributedCache::Clear() {
  for (auto &item : client_pool_) {
    item->Disconnect();
  }
  client_pool_.clear();
}
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
