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

#ifndef MINDSPORE_CCSRC_FL_SERVER_BUFFER_STATUS_H
#define MINDSPORE_CCSRC_FL_SERVER_BUFFER_STATUS_H
#include <string>

namespace mindspore {
namespace fl {
namespace cache {
enum CacheStatusCode {
  kCacheSuccess = 0,
  kCacheNil = 1,
  kCacheExist = 2,
  kCacheNetErr = 3,
  kCacheInnerErr = 4,
  kCacheTypeErr = 5,
  kCacheParamFailed = 6,
};

class CacheStatus {
 public:
  CacheStatus() = default;
  CacheStatus(CacheStatusCode code, const std::string &detail = "")  // NOLINT(runtime/explicit)
      : code_(code), detail_(detail) {}
  bool IsSuccess() const { return code_ == kCacheSuccess; }
  bool IsNil() const { return code_ == kCacheNil; }
  CacheStatusCode GetCode() const { return code_; }
  std::string GetDetail() const { return detail_; }
  bool operator==(CacheStatusCode other) const { return code_ == other; }
  bool operator!=(CacheStatusCode other) const { return code_ != other; }
  bool operator==(const CacheStatus &other) const { return code_ == other.code_; }
  bool operator!=(const CacheStatus &other) const { return code_ != other.code_; }
  operator bool() = delete;

 private:
  CacheStatusCode code_ = kCacheInnerErr;
  std::string detail_;
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FL_SERVER_BUFFER_STATUS_H
