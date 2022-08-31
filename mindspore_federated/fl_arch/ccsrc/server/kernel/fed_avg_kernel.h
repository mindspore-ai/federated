/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_FED_AVG_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_FED_AVG_KERNEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include <map>
#include "common/common.h"
#include "server/collective_ops_impl.h"
#include "server/local_meta_store.h"
#include "server/executor.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
// The implementation for the federated average. We do weighted average for the weights. The uploaded weights from
// FL-clients is already multiplied by its data size so only sum and division are done in this kernel.

// Pay attention that this kernel is the distributed version of federated average, which means each server node in the
// cluster in invalved in the aggragation process. So the DistributedCountService and CollectiveOpsImpl are called.
template <typename T, typename S>
class FedAvgKernel {
 public:
  static bool AllReduce(const std::map<std::string, std::string> &server_map, ParamAggregationInfo *info) {
    if (info == nullptr) {
      return false;
    }
    T *weight_addr = reinterpret_cast<T *>(info->weight_data);
    if (!CollectiveOpsImpl::GetInstance().AllReduce<T>(info->name, weight_addr, weight_addr,
                                                       info->weight_size / sizeof(T), server_map)) {
      MS_LOG(ERROR) << "Federated average allreduce failed.";
      return false;
    }
    if (!CollectiveOpsImpl::GetInstance().AllReduce<S>(info->name + "_data_size", &info->data_size, &info->data_size, 1,
                                                       server_map)) {
      MS_LOG(ERROR) << "Federated average allreduce failed.";
      return false;
    }
    auto data_size = info->data_size;
    if (data_size == 0) {
      MS_LOG(INFO) << "Parameter:" << info->name << " data size is 0, do not need to run fed avg.";
      return false;
    }
    LocalMetaStore::GetInstance().put_value(kCtxFedAvgTotalDataSize, data_size);
    auto elem_num = info->weight_size / sizeof(T);
    for (size_t i = 0; i < elem_num; i++) {
      weight_addr[i] /= data_size;
    }
    return true;
  }

  static void Launch(const Address &update_weight, size_t update_data_size, ParamAggregationInfo *info) {
    if (info == nullptr) {
      return;
    }
    // The weight and new_weight values should be multiplied by clients already, so we don't need to do multiplication
    // again.
    auto weight_addr = reinterpret_cast<T *>(info->weight_data);
    auto new_weight_addr = reinterpret_cast<const T *>(update_weight.addr);

    auto elem_num = info->weight_size / sizeof(T);
    for (size_t i = 0; i < elem_num; i++) {
      weight_addr[i] += new_weight_addr[i];
    }
    info->data_size += update_data_size;
  }
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_FED_AVG_KERNEL_H_
