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
#include "vertical/utils/data_join_utils.h"
#include <vector>
#include <string>

namespace mindspore {
namespace fl {
void CreateWorkerRegisterProto(datajoin::WorkerRegisterProto *workerRegisterProto,
                               const WorkerRegisterItemPy &workerRegisterItemPy) {
  workerRegisterProto->set_worker_name(workerRegisterItemPy.worker_name());
}

void CreateWorkerConfigProto(datajoin::WorkerConfigProto *workerConfigProto,
                             const WorkerConfigItemPy &workerConfigItemPy) {
  workerConfigProto->set_primary_key(workerConfigItemPy.primary_key());
  workerConfigProto->set_bucket_num(workerConfigItemPy.bucket_num());
  workerConfigProto->set_shard_num(workerConfigItemPy.shard_num());
  workerConfigProto->set_join_type(workerConfigItemPy.join_type());
}

WorkerRegisterItemPy ParseWorkerRegisterProto(const datajoin::WorkerRegisterProto &workerRegisterProto) {
  WorkerRegisterItemPy workerRegisterItem;
  workerRegisterItem.set_worker_name(workerRegisterProto.worker_name());
  return workerRegisterItem;
}

WorkerConfigItemPy ParseWorkerConfigProto(const datajoin::WorkerConfigProto &workerConfigProto) {
  WorkerConfigItemPy workerConfigItemPy;
  workerConfigItemPy.set_primary_key(workerConfigProto.primary_key());
  workerConfigItemPy.set_bucket_num(workerConfigProto.bucket_num());
  workerConfigItemPy.set_shard_num(workerConfigProto.shard_num());
  workerConfigItemPy.set_join_type(workerConfigProto.join_type());
  MS_LOG(INFO) << "workerConfigItemPy, primary_key is " << workerConfigItemPy.primary_key();
  MS_LOG(INFO) << "workerConfigItemPy, bucket_num is " << workerConfigItemPy.bucket_num();
  MS_LOG(INFO) << "workerConfigItemPy, join_type is " << workerConfigItemPy.join_type();
  return workerConfigItemPy;
}
}  // namespace fl
}  // namespace mindspore
