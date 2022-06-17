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
#ifndef MINDSPORE_CCSRC_FL_DISTRIBUTED_CACHE_H
#define MINDSPORE_CCSRC_FL_DISTRIBUTED_CACHE_H

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <map>
#include "distributed_cache/cache_status.h"

namespace mindspore {
namespace fl {
namespace cache {
struct DistributedCacheConfig {
  std::string type;
  std::string address;
  std::string plugin_lib_path;
  bool enable_ssl = false;
  std::unordered_map<std::string, std::string> configs;
};

class RedisClientBase {
 public:
  RedisClientBase() = default;
  RedisClientBase(const RedisClientBase &other) = delete;
  virtual ~RedisClientBase() = default;

  virtual bool IsValid() = 0;
  virtual void Disconnect() = 0;
  virtual CacheStatus Connect(bool retry_connect) = 0;
  virtual CacheStatus Reconnect() = 0;
  // Del
  virtual CacheStatus Del(const std::vector<std::string> &keys) = 0;
  CacheStatus Del(const std::string &key) { return Del(std::vector<std::string>({key})); }
  // expire
  virtual CacheStatus Expire(const std::string &key, uint64_t seconds) = 0;
  // set operator
  virtual CacheStatus SAdd(const std::string &key, const std::string &member) = 0;
  virtual CacheStatus SIsMember(const std::string &key, const std::string &member, bool *value) = 0;
  virtual CacheStatus SMembers(const std::string &key, std::vector<std::string> *members) = 0;
  // hash operator
  virtual CacheStatus HExists(const std::string &key, const std::string &filed, bool *value) = 0;
  virtual CacheStatus HSet(const std::string &key, const std::string &filed, const std::string &value) = 0;
  virtual CacheStatus HSetNx(const std::string &key, const std::string &filed, const std::string &value) = 0;
  virtual CacheStatus HMSet(const std::string &key, const std::unordered_map<std::string, std::string> &items) = 0;
  virtual CacheStatus HGet(const std::string &key, const std::string &filed, std::string *value) = 0;
  virtual CacheStatus HGetAll(const std::string &key, std::unordered_map<std::string, std::string> *items) = 0;
  virtual CacheStatus HIncr(const std::string &key, const std::string &filed, uint64_t *new_value) = 0;
  virtual CacheStatus HDel(const std::string &key, const std::string &filed) = 0;
  // string operator
  virtual CacheStatus Get(const std::string &key, std::string *value) = 0;
  virtual CacheStatus SetEx(const std::string &key, const std::string &value, uint64_t seconds) = 0;
  virtual CacheStatus SetNx(const std::string &key, const std::string &value) = 0;
  virtual CacheStatus SetExNx(const std::string &key, const std::string &value, uint64_t seconds) = 0;
  virtual CacheStatus Incr(const std::string &key, uint64_t *new_value) = 0;

  // Set Hash filed and int64 value
  CacheStatus HMSet(const std::string &key, const std::unordered_map<std::string, uint64_t> &items);
  // Get Hash filed and parse to int64
  CacheStatus HGetAll(const std::string &key, std::unordered_map<std::string, uint64_t> *items);
  // Get Hash filed and parse to int64
  CacheStatus HGet(const std::string &key, const std::string &filed, uint64_t default_val, uint64_t *value);
  // Get String value and parse to int64
  CacheStatus Get(const std::string &key, uint64_t default_val, uint64_t *value);
};

class DistributedCacheBase {
 public:
  DistributedCacheBase() = default;
  virtual ~DistributedCacheBase() = default;
  virtual bool Init(const DistributedCacheConfig &cache_config, int64_t timeout) = 0;
  virtual std::shared_ptr<RedisClientBase> GetOneClient() = 0;
  virtual bool HasInvalid() const = 0;
  virtual CacheStatus RetryConnect() = 0;
};

class DistributedCacheLoader {
 public:
  static DistributedCacheLoader &Instance() {
    static DistributedCacheLoader instance;
    return instance;
  }
  bool InitCacheImpl(const DistributedCacheConfig &cache_config);
  std::shared_ptr<RedisClientBase> GetOneClient();
  bool HasInvalid() const;
  CacheStatus RetryConnect();

  void set_available(bool available) { available_ = available; }
  bool available() const { return available_; }

 private:
  std::shared_ptr<DistributedCacheBase> cache_impl_ = nullptr;
  bool available_ = true;
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_DISTRIBUTED_CACHE_H
