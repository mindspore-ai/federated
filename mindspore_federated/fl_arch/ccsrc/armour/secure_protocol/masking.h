/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_ARMOUR_RANDOM_H
#define MINDSPORE_ARMOUR_RANDOM_H

#include <random>
#include <vector>
#include "armour/secure_protocol/encrypt.h"

namespace mindspore {
namespace fl {
namespace armour {
class MS_EXPORT Masking {
 public:
  static int GetMasking(std::vector<float> *noise, int noise_len, const uint8_t *secret, int secret_len,
                        const uint8_t *ivec, int ivec_size);
};
}  // namespace armour
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_ARMOUR_RANDOM_H
