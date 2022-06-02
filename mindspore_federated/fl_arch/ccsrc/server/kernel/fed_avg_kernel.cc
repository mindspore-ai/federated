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

#include "server/kernel/fed_avg_kernel.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
REG_AGGREGATION_KERNEL_TWO(FedAvg,
                           ParamsInfo()
                             .AddInputNameType(kWeight, kNumberTypeFloat32)
                             .AddInputNameType(kDataSize, kNumberTypeUInt64)
                             .AddInputNameType(kNewWeight, kNumberTypeFloat32)
                             .AddInputNameType(kNewDataSize, kNumberTypeUInt64),
                           FedAvgKernel, float, size_t)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore