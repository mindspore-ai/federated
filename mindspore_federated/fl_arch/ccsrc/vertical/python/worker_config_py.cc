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
#include "vertical/python/worker_config_py.h"
#include <functional>
#include <vector>

namespace mindspore {
namespace fl {
void WorkerConfigItemPy::set_primary_key(const std::string &primary_key) { primary_key_ = primary_key; }
std::string WorkerConfigItemPy::primary_key() const { return primary_key_; }

void WorkerConfigItemPy::set_bucket_num(const uint64_t &bucket_num) { bucket_num_ = bucket_num; }
uint64_t WorkerConfigItemPy::bucket_num() const { return bucket_num_; }

void WorkerConfigItemPy::set_shard_num(const uint64_t &shard_num) { shard_num_ = shard_num; }
uint64_t WorkerConfigItemPy::shard_num() const { return shard_num_; }

void WorkerConfigItemPy::set_join_type(const std::string &join_type) { join_type_ = join_type; }
std::string WorkerConfigItemPy::join_type() const { return join_type_; }
}  // namespace fl
}  // namespace mindspore
