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

#ifndef MINDSPORE_FEDERATED_COMMON_YAML_CONFIG_H
#define MINDSPORE_FEDERATED_COMMON_YAML_CONFIG_H
#include <string>
#include <unordered_map>
#include <functional>
#include <vector>
#include "common/status.h"
#include "common/constants.h"

namespace mindspore {
namespace fl {
namespace yaml {
enum YamlValType {
  kYamlInt = 0,
  kYamlFloat = 1,
  kYamlBool = 2,
  kYamlStr = 3,
  kYamlDict = 4,
};

struct YamlConfigItem {
  YamlValType type = kYamlInt;
  int64_t int_val = 0;
  float float_val = 0;
  bool bool_val = false;
  std::string str_val;
};

struct RoundConfig {
  uint64_t start_fl_job_threshold = 0;
  uint64_t start_fl_job_time_window = 0;
  float update_model_ratio = 0;
  uint64_t update_model_time_window = 0;
  uint64_t global_iteration_time_window = 0;
};

struct WorkerConfig {
  uint64_t step_num_per_iteration = 0;
};

struct SummaryConfig {
  std::string participation_time_level = "5,15";
  uint64_t continuous_failure_times = 10;
  std::string metrics_file = "metrics.json";
  std::string failure_event_file = "event.txt";
  std::string data_rate_dir = "..";
};

enum CompareOp {
  CMP_NONE = 0,
  EQ = 1,  // ==
  NE = 2,  // !=
  LT = 3,  // <
  LE = 4,  // <=
  GT = 5,  // >
  GE = 6,  // >=
};

enum RangeOp {
  INC_NONE,
  // scalar range check
  INC_NEITHER,  // (), include neither
  INC_LEFT,     // [), include left
  INC_RIGHT,    // (], include right
  INC_BOTH,     // [], include both
};

template <class T>
struct CheckNum {
 public:
  CheckNum() = default;
  CheckNum(T min, T max, RangeOp range_op) : min_(min), max_(max), range_op_(range_op) {}
  CheckNum(T val, CompareOp cmp_op) : val_(val), cmp_op_(cmp_op) {}

  FlStatus CheckRange(T num) {
    if (range_op_ == INC_NEITHER && !(num > min_ && num < max_)) {  // (min, max)
      return {kFlFailed, "expect value to be range of (" + std::to_string(min_) + "," + std::to_string(max_) +
                           "), but got " + std::to_string(num)};
    } else if (range_op_ == INC_LEFT && !(num >= min_ && num < max_)) {  // [min, max)
      return {kFlFailed, "value is expected to be range of [" + std::to_string(min_) + "," + std::to_string(max_) +
                           "), but got " + std::to_string(num)};
    } else if (range_op_ == INC_RIGHT && !(num > min_ && num <= max_)) {  // (min, max]
      return {kFlFailed, "value is expected value to be range of (" + std::to_string(min_) + "," +
                           std::to_string(max_) + "], but got " + std::to_string(num)};
    } else if (range_op_ == INC_BOTH && !(num >= min_ && num <= max_)) {  // [min, max]
      return {kFlFailed, "value is expected to be range of [" + std::to_string(min_) + "," + std::to_string(max_) +
                           "], but got " + std::to_string(num)};
    }
    return kFlSuccess;
  }
  FlStatus CheckCmp(T num) {
    if (cmp_op_ == GE && !(num >= val_)) {  // >=
      return {kFlFailed, "value is expected >=" + std::to_string(val_) + ", but got " + std::to_string(num)};
    } else if (cmp_op_ == GT && !(num > val_)) {  //
      return {kFlFailed, "value is expected >" + std::to_string(val_) + ", but got " + std::to_string(num)};
    } else if (cmp_op_ == LE && !(num <= val_)) {  // <=
      return {kFlFailed, "value is expected <=" + std::to_string(val_) + ", but got " + std::to_string(num)};
    } else if (cmp_op_ == LT && !(num < val_)) {  // <
      return {kFlFailed, "value is expected <" + std::to_string(val_) + ", but got " + std::to_string(num)};
    }
    return kFlSuccess;
  }
  FlStatus Check(T num) {
    auto status = CheckRange(num);
    if (!status.IsSuccess()) {
      return status;
    }
    status = CheckCmp(num);
    if (!status.IsSuccess()) {
      return status;
    }
    return kFlSuccess;
  }

 private:
  T min_ = 0;
  T max_ = 0;
  RangeOp range_op_ = INC_NONE;
  T val_ = 0;
  CompareOp cmp_op_ = CMP_NONE;
};

using CheckInt = CheckNum<int64_t>;
using CheckFloat = CheckNum<float>;

class YamlConfig {
 public:
  void Load(const std::unordered_map<std::string, YamlConfigItem> &items, const std::string &yaml_config_file,
            const std::string &role, bool enable_ssl);

 private:
  bool Get(const std::string &key, std::string *value, bool required,
           const std::vector<std::string> &choices = {}) const;
  bool Get(const std::string &key, uint64_t *value, bool required, CheckInt check) const;
  bool Get(const std::string &key, float *value, bool required, CheckFloat check) const;
  bool Get(const std::string &key, bool *value, bool required) const;

  void Get(const std::string &key, const std::function<void(const std::string &)> &set_fun, bool required,
           const std::vector<std::string> &choices = {}) const;
  void Get(const std::string &key, const std::function<void(uint64_t)> &set_fun, bool required, CheckInt check) const;
  void Get(const std::string &key, const std::function<void(float)> &set_fun, bool required, CheckFloat check) const;
  void Get(const std::string &key, const std::function<void(bool)> &set_fun, bool required) const;

  void InitCommonConfig();
  void InitDistributedCacheConfig();
  void InitRoundConfig();
  void InitSummaryConfig();
  void InitEncryptConfig();
  void InitCompressionConfig();
  void InitSslConfig();
  void InitClientVerifyConfig();
  void InitClientConfig();
  void CheckYamlConfig();

  std::unordered_map<std::string, YamlConfigItem> items_;
  std::string yaml_config_file_;
  bool enable_ssl_ = false;
};
}  // namespace yaml
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FEDERATED_COMMON_YAML_CONFIG_H
