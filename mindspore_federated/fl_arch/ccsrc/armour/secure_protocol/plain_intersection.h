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

#ifndef MINDSPORE_FEDERATED_PLAIN_INTERSECTION_H
#define MINDSPORE_FEDERATED_PLAIN_INTERSECTION_H

#include <vector>
#include <string>

#include "armour/secure_protocol/psi.h"
#include "armour/base_crypto/hash.h"
#include "armour/util/io_util.h"
#include "vertical/vertical_server.h"

namespace mindspore {
namespace fl {
namespace psi {
MS_EXPORT std::vector<std::string> PlainIntersection(const std::vector<std::string> &input_vct,
                                                     const std::string &comm_role, size_t thread_num, size_t bin_id,
                                                     const std::string &target_server_name);
}  // namespace psi
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_FEDERATED_PLAIN_INTERSECTION_H
