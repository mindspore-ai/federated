/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_COMPRESSION_COMPRESS_COMMON_H_
#define MINDSPORE_CCSRC_FL_COMPRESSION_COMPRESS_COMMON_H_

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>
#include "common/utils/log_adapter.h"
#include "common/utils/visible.h"
#include "vertical/python/tensor_py.h"

namespace mindspore {
namespace fl {
namespace compression {

const int k128 = 128;
const int k64 = 64;
const int k32 = 32;
const int k16 = 16;
const int k8 = 8;
const int k7 = 7;
const int k4 = 4;
const int k2 = 2;
const int k1 = 1;

const float kEps = 1e-10f;

constexpr auto kNoCompress = "no_compress";
constexpr auto kMinMax = "min_max";
constexpr auto kBitPack = "bit_pack";

}  // namespace compression
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMPRESSION_COMPRESS_COMMON_H_
