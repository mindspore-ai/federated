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

#ifndef MINDSPORE_FEDERATED_HASH_H
#define MINDSPORE_FEDERATED_HASH_H

#include <openssl/sha.h>

#include <vector>
#include <string>

#include "common/parallel_for.h"
#include "armour/base_crypto/base_unit.h"

namespace mindspore {
namespace fl {
namespace psi {

std::string HashInput(const std::string &item);

std::vector<std::string> HashInputs(const std::vector<std::string> &items, size_t thread_num, size_t chunk_size);

}  // namespace psi
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_FEDERATED_HASH_H
