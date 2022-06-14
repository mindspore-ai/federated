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
#ifndef MINDSPORE_CCSRC_FL_DISTRIBUTED_CACHE_COMMON_H
#define MINDSPORE_CCSRC_FL_DISTRIBUTED_CACHE_COMMON_H
#include <string>
#include <unordered_map>
#include "common/common.h"

namespace mindspore {
namespace fl {
namespace cache {
static inline bool Str2Int64(const std::string &str, int64_t *value) {
  try {
    *value = std::stoll(str);
  } catch (const std::invalid_argument &e) {
    return false;
  } catch (const std::out_of_range &e) {
    return false;
  }
  return true;
}
static inline bool Str2Uint64(const std::string &str, uint64_t *value) {
  try {
    auto val = std::stoll(str);
    if (val < 0) {
      return false;
    }
    *value = static_cast<uint64_t>(val);
  } catch (const std::invalid_argument &e) {
    return false;
  } catch (const std::out_of_range &e) {
    return false;
  }
  return true;
}
static inline bool GetIntValue(const std::unordered_map<std::string, std::string> &items, const std::string &key,
                               int64_t *value) {
  auto it = items.find(key);
  if (it == items.end()) {
    return false;
  }
  return Str2Int64(it->second, value);
}
static inline bool GetUintValue(const std::unordered_map<std::string, std::string> &items, const std::string &key,
                                uint64_t *value) {
  auto it = items.find(key);
  if (it == items.end()) {
    return false;
  }
  return Str2Uint64(it->second, value);
}
static inline bool GetStrValue(const std::unordered_map<std::string, std::string> &items, const std::string &key,
                               std::string *value) {
  auto it = items.find(key);
  if (it == items.end()) {
    return false;
  }
  *value = it->second;
  return true;
}

class DistributedCacheUnavailable : public std::exception {
 public:
  DistributedCacheUnavailable() = default;
  ~DistributedCacheUnavailable() = default;
  const char *what() const noexcept override { return "Distributed cache server is unavailable."; }
};

#define THROW_CACHE_UNAVAILABLE throw DistributedCacheUnavailable()
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_DISTRIBUTED_CACHE_COMMON_H
