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

#ifndef MINDSPORE_CCSRC_FL_REDIS_H
#define MINDSPORE_CCSRC_FL_REDIS_H

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <map>
#include <atomic>

#include "hiredis/hiredis.h"
#include "hiredis/hiredis_ssl.h"
#include "distributed_cache/cache_status.h"
#include "distributed_cache/distributed_cache.h"

namespace mindspore {
namespace fl {
namespace cache {
class RedisReply {
 public:
  RedisReply() = default;
  explicit RedisReply(redisReply *reply) noexcept;
  RedisReply(const RedisReply &other) = delete;
  RedisReply &operator=(const RedisReply &other) = delete;

  RedisReply(RedisReply &&other) noexcept;
  RedisReply &operator=(RedisReply &&other) noexcept;

  int GetType() const;
  bool IsValid() const;
  std::string GetError() const;
  bool GetStatus(std::string *value) const;
  bool GetInteger(uint64_t *value) const;
  bool GetString(std::string *value) const;
  bool GetArray(std::vector<std::string> *value) const;
  bool GetMap(std::unordered_map<std::string, std::string> *value) const;

  bool IsNil() const;

 private:
  struct RedisReplyDeleter {
    void operator()(redisReply *reply) {
      if (reply != nullptr) {
        freeReplyObject(reply);
      }
    }
  };
  std::unique_ptr<redisReply, RedisReplyDeleter> redis_reply_;
};

struct RedisSSLConfig {
  std::string cacert_filename;
  std::string capath;
  std::string cert_filename;
  std::string private_key_filename;
  std::string server_name;
};

class RedisClient : public RedisClientBase {
 public:
  explicit RedisClient(const std::string &server_address, redisSSLContext *ssl_context, int64_t timeout);
  RedisClient(const RedisClient &other) = delete;
  RedisClient(RedisClient &&other) = delete;
  ~RedisClient();

  bool IsValid() override;
  void Disconnect() override;
  CacheStatus Connect() override;
  CacheStatus Reconnect() override;
  // Del
  CacheStatus Del(const std::vector<std::string> &keys) override;
  // expire
  CacheStatus Expire(const std::string &key, uint64_t seconds) override;
  // set operator
  CacheStatus SAdd(const std::string &key, const std::string &member) override;
  CacheStatus SIsMember(const std::string &key, const std::string &member, bool *value) override;
  CacheStatus SMembers(const std::string &key, std::vector<std::string> *members) override;
  // hash operator
  CacheStatus HExists(const std::string &key, const std::string &filed, bool *bool_value) override;
  CacheStatus HSet(const std::string &key, const std::string &filed, const std::string &value) override;
  CacheStatus HSetNx(const std::string &key, const std::string &filed, const std::string &value) override;
  CacheStatus HMSet(const std::string &key, const std::unordered_map<std::string, std::string> &items) override;
  CacheStatus HGet(const std::string &key, const std::string &filed, std::string *value) override;
  CacheStatus HGetAll(const std::string &key, std::unordered_map<std::string, std::string> *items) override;
  CacheStatus HIncr(const std::string &key, const std::string &filed, uint64_t *new_value) override;
  CacheStatus HDel(const std::string &key, const std::string &filed) override;
  //
  CacheStatus Get(const std::string &key, std::string *value) override;
  CacheStatus SetEx(const std::string &key, const std::string &value, uint64_t seconds) override;
  CacheStatus SetNx(const std::string &key, const std::string &value) override;
  CacheStatus SetExNx(const std::string &key, const std::string &value, uint64_t seconds) override;
  CacheStatus Incr(const std::string &key, uint64_t *new_value) override;

 protected:
  CacheStatus ReconnectInner();
  RedisReply RunCommand(int argc, const char **argv, const size_t *argvlen);
  RedisReply RunCommand(const std::vector<std::string> &args);
  RedisReply Eval(const std::string &script, const std::vector<std::string> &keys,
                  const std::vector<std::string> &args);

  bool IsUnixAddress(const std::string &server_address);

  std::mutex lock_;
  std::string server_address_;
  redisSSLContext *ssl_context_ = nullptr;
  redisContext *redis_context_ = nullptr;
  int64_t timeout_ = 0;
};

class RedisDistributedCache : public DistributedCacheBase {
 public:
  RedisDistributedCache() = default;
  ~RedisDistributedCache();
  bool Init(const DistributedCacheConfig &cache_config, int64_t timeout) override;
  std::shared_ptr<RedisClientBase> GetOneClient() override;
  bool HasInvalid() const override;
  CacheStatus RetryConnect() override;

 private:
  DistributedCacheConfig cache_config_;
  static constexpr uint32_t thread_pool_size_ = 4;
  std::atomic_uint64_t cur_client_ret_index_ = 0;

  redisSSLContext *ssl_context_ = nullptr;
  std::vector<std::shared_ptr<RedisClient>> client_pool_;

  CacheStatus ParseSSLConfig(const std::unordered_map<std::string, std::string> &configs, RedisSSLConfig *ssl_config_p);
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_REDIS_H
