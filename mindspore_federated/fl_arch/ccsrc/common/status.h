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

#ifndef MINDSPORE_CCSRC_FL_SERVER_COMMON_STATUS_H
#define MINDSPORE_CCSRC_FL_SERVER_COMMON_STATUS_H
#include <string>

namespace mindspore {
namespace fl {
enum FlStatusCode {
  kFlSuccess = 0,
  kFlFailed = 1,
  kSystemError = 2,
  kInvalidInputs = 3,
  kRequestError = 4,
  kNotReadyError = 5,
  kAggregationNotDone = 6,
  kTypeError = 7,
};

class FlStatus {
 public:
  FlStatus() = default;
  FlStatus(FlStatusCode code, const std::string &detail = "")  // NOLINT(runtime/explicit)
      : code_(code), detail_(detail) {}
  FlStatus(const FlStatus &other) {
    code_ = other.code_;
    detail_ = other.detail_;
  }
  bool IsSuccess() const { return code_ == kFlSuccess; }
  FlStatusCode GetCode() const { return code_; }
  std::string StatusMessage() const { return detail_; }
  bool operator==(FlStatusCode other) const { return code_ == other; }
  bool operator!=(FlStatusCode other) const { return code_ != other; }
  operator bool() = delete;

 private:
  FlStatusCode code_ = kFlFailed;
  std::string detail_;
};
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FL_SERVER_COMMON_STATUS_H
