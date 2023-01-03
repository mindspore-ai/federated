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

#include "armour/base_crypto/hash.h"

namespace mindspore {
namespace fl {
namespace psi {

std::string HashInput(const std::string &item) {
  std::string hash_result(LENGTH_32, '\0');
  SHA256((const uint8_t *)item.data(), item.size(), (uint8_t *)hash_result.data());
  return hash_result;
}

std::vector<std::string> HashInputs(const std::vector<std::string> *items, size_t thread_num, size_t chunk_size) {
  time_t time_start;
  time_t time_end;
  time(&time_start);
  std::vector<std::string> ret(items->size());

  ParallelSync parallel_sync(thread_num);
  parallel_sync.parallel_for(0, ret.size(), chunk_size, [&](size_t beg, size_t end) {
    for (size_t i = beg; i < end; i++) {
      ret[i] = HashInput(items->at(i));
    }
  });

  time(&time_end);
  MS_LOG(INFO) << "Thread num is " << parallel_sync.get_thread_num() << ", Task num is " << parallel_sync.get_task_num()
               << ", HashInputs time cost: " << difftime(time_end, time_start) << " s.";
  return ret;
}

}  // namespace psi
}  // namespace fl
}  // namespace mindspore
