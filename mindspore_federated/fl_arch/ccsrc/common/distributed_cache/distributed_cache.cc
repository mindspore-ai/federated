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
#include "distributed_cache/distributed_cache.h"
#include "common/common.h"
#include "distributed_cache/redis/redis.h"
#include "distributed_cache/redis_keys.h"
#include "distributed_cache/instance_context.h"
#include "distributed_cache/common.h"

namespace mindspore {
namespace fl {
namespace cache {
bool DistributedCacheLoader::InitCacheImpl(const DistributedCacheConfig &cache_config) {
  if (cache_impl_ != nullptr) {
    MS_LOG_ERROR << "InitCacheImpl should not be init twice";
    return true;
  }
  auto cache_impl = std::make_shared<RedisDistributedCache>();
  constexpr int64_t cache_timeout_in_secs = 5 * 60;  // 5 min
  if (!cache_impl->Init(cache_config, cache_timeout_in_secs)) {
    return false;
  }
  cache_impl_ = cache_impl;
  return true;
}

std::shared_ptr<RedisClientBase> DistributedCacheLoader::GetOneClient() {
  if (cache_impl_ == nullptr) {
    MS_LOG_ERROR << "GetOneClient should called after InitCacheImpl";
    return nullptr;
  }
  return cache_impl_->GetOneClient();
}

bool DistributedCacheLoader::HasInvalid() const {
  if (cache_impl_ == nullptr) {
    return false;
  }
  return cache_impl_->HasInvalid();
}

CacheStatus DistributedCacheLoader::RetryConnect() {
  if (cache_impl_ == nullptr) {
    return {kCacheInnerErr, "cache_impl_ is nullptr"};
  }
  return cache_impl_->RetryConnect();
}

CacheStatus RedisClientBase::HMSet(const std::string &key, const std::unordered_map<std::string, uint64_t> &items) {
  std::unordered_map<std::string, std::string> items_str;
  for (auto &item : items) {
    items_str[item.first] = std::to_string(item.second);
  }
  return HMSet(key, items_str);
}

CacheStatus RedisClientBase::HGetAll(const std::string &key, std::unordered_map<std::string, uint64_t> *items) {
  if (items == nullptr) {
    return kCacheInnerErr;
  }
  std::unordered_map<std::string, std::string> items_str;
  auto status = HGetAll(key, &items_str);
  if (!status.IsSuccess()) {
    return status;
  }
  std::unordered_map<std::string, uint64_t> items_ret;
  for (auto &item : items_str) {
    uint64_t item_value;
    if (!Str2Uint64(item.second, &item_value)) {
      MS_LOG_WARNING << "Expect hash filed value to be int, key: " << key << ", filed: " << item.first;
      return kCacheTypeErr;
    }
    items_ret[item.first] = item_value;
  }
  *items = std::move(items_ret);
  return kCacheSuccess;
}

// Get Hash filed and parse to int64
CacheStatus RedisClientBase::HGet(const std::string &key, const std::string &filed, uint64_t default_val,
                                  uint64_t *value) {
  if (value == nullptr) {
    return kCacheInnerErr;
  }
  std::string value_str;
  auto status = HGet(key, filed, &value_str);
  if (status == kCacheNil) {
    *value = default_val;
    return kCacheSuccess;
  }
  if (!status.IsSuccess()) {
    return status;
  }
  if (!Str2Uint64(value_str, value)) {
    MS_LOG_WARNING << "Expect hash filed value to be int, key: " << key << ", filed: " << filed;
    return kCacheTypeErr;
  }
  return kCacheSuccess;
}
// Get String value and parse to int64
CacheStatus RedisClientBase::Get(const std::string &key, uint64_t default_val, uint64_t *value) {
  if (value == nullptr) {
    return kCacheInnerErr;
  }
  std::string value_str;
  auto status = Get(key, &value_str);
  if (status == kCacheNil) {
    *value = default_val;
    return kCacheSuccess;
  }
  if (!status.IsSuccess()) {
    return status;
  }
  if (!Str2Uint64(value_str, value)) {
    MS_LOG_WARNING << "Expect string value to be int, key: " << key;
    return kCacheTypeErr;
  }
  return kCacheSuccess;
}
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
