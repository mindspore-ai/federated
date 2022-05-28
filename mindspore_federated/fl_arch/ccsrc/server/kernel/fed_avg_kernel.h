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
#include "common/common.h"
#include "server/collective_ops_impl.h"
#include "server/distributed_count_service.h"
#include "server/local_meta_store.h"
#include "server/kernel/aggregation_kernel.h"
#include "server/kernel/aggregation_kernel_factory.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
constexpr size_t kFedAvgInputsNum = 4;
// The implementation for the federated average. We do weighted average for the weights. The uploaded weights from
// FL-clients is already multiplied by its data size so only sum and division are done in this kernel.

// Pay attention that this kernel is the distributed version of federated average, which means each server node in the
// cluster in invalved in the aggragation process. So the DistributedCountService and CollectiveOpsImpl are called.
template <typename T, typename S>
class FedAvgKernel : public AggregationKernelMod {
 public:
  FedAvgKernel() :
        weight_addr_(nullptr),
        data_size_addr_(nullptr),
        new_weight_addr_(nullptr),
        new_data_size_addr_(nullptr) {}
  ~FedAvgKernel() override = default;

  void InitKernel(const std::string &param_name) override {
    auto &feature_maps = FLContext::instance()->feature_maps();
    const auto& feature_map = feature_maps[param_name];

    size_t weight_size = feature_map.weight_size;
    size_t new_weight_size = weight_size;

    input_size_list_.push_back(weight_size);
    input_size_list_.push_back(sizeof(size_t));
    input_size_list_.push_back(new_weight_size);
    input_size_list_.push_back(sizeof(size_t));

    MS_LOG(DEBUG) <<"weight size is " << feature_map.weight_size << ", weight shape is " << feature_map.weight_shape
                  << ", weight type is " << feature_map.weight_type;

    LocalMetaStore::GetInstance().put_aggregation_feature_map(param_name, feature_map);
    MS_LOG(INFO) << "Aggregate Weight full name is " << param_name;
    return;
  }

  bool AllReduce() override {
    std::unique_lock<std::mutex> lock(weight_mutex_);
    MS_ERROR_IF_NULL_W_RET_VAL(weight_addr_, false);
    MS_ERROR_IF_NULL_W_RET_VAL(data_size_addr_, false);
    MS_ERROR_IF_NULL_W_RET_VAL(weight_addr_->addr, false);
    MS_ERROR_IF_NULL_W_RET_VAL(data_size_addr_->addr, false);
    T *weight_addr = reinterpret_cast<T *>(weight_addr_->addr);
    size_t weight_size = weight_addr_->size;
    S *data_size_addr = reinterpret_cast<S *>(data_size_addr_->addr);
    if (!CollectiveOpsImpl::GetInstance().AllReduce<T>(name_, weight_addr, weight_addr, weight_size / sizeof(T))) {
      MS_LOG(ERROR) << "Federated average allreduce failed.";
      return false;
    }
    if (!CollectiveOpsImpl::GetInstance().AllReduce<S>(name_ + "_data_size", data_size_addr, data_size_addr, 1)) {
      MS_LOG(ERROR) << "Federated average allreduce failed.";
      return false;
    }
    if (data_size_addr[0] == 0) {
      MS_LOG(ERROR) << "After AllReduce, the data size is 0.";
      return false;
    }
    LocalMetaStore::GetInstance().put_value(kCtxFedAvgTotalDataSize, data_size_addr[0]);
    for (size_t i = 0; i < weight_size / sizeof(T); i++) {
      weight_addr[i] /= data_size_addr[0];
    }
    done_ = true;
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    if (inputs.size() != kFedAvgInputsNum) {
      MS_LOG(ERROR) << "The inputs number of FedAvgKernel should be 4, but got " << inputs.size();
      return false;
    }
    for (size_t i = 0; i < inputs.size(); i++) {
      MS_ERROR_IF_NULL_W_RET_VAL(inputs[i]->addr, false);
    }

    std::unique_lock<std::mutex> lock(weight_mutex_);
    if (done_) {
      MS_LOG(INFO) << "AllReduce for " << name_ << " has finished";
      return true;
    }
    // The weight and new_weight values should be multiplied by clients already, so we don't need to do multiplication
    // again.
    T *weight_addr = reinterpret_cast<T *>(inputs[0]->addr);
    S *data_size_addr = reinterpret_cast<S *>(inputs[1]->addr);
    T *new_weight_addr = reinterpret_cast<T *>(inputs[2]->addr);
    S *new_data_size_addr = reinterpret_cast<S *>(inputs[3]->addr);

    MS_LOG(DEBUG) << "Iteration: " << LocalMetaStore::GetInstance().curr_iter_num() << " launching FedAvgKernel for "
                  << name_ << " new data size is " << new_data_size_addr[0] << ", current total data size is "
                  << data_size_addr[0];
    for (size_t i = 0; i < inputs[2]->size / sizeof(T); i++) {
      weight_addr[i] += new_weight_addr[i];
    }
    data_size_addr[0] += new_data_size_addr[0];
    lock.unlock();

    accum_count_++;
    return true;
  }

  void Reset() override {
    accum_count_ = 0;
    done_ = false;
    ClearWeightAndDataSize();
  }

  bool IsAggregationDone() override { return done_; }

  void SetParameterAddress(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                           const std::vector<AddressPtr> &outputs) override {
    weight_addr_ = inputs[0];
    data_size_addr_ = inputs[1];
    new_weight_addr_ = inputs[2];
    new_data_size_addr_ = inputs[3];
    return;
  }
  bool ReInitForUpdatingHyperParams(size_t aggr_threshold) override {
    done_count_ = aggr_threshold;
    return true;
  }

 private:
  // In some cases, the Launch method is not called and the weights involved in AllReduce should be set to 0.
  void ClearWeightAndDataSize() {
    MS_ERROR_IF_NULL_WO_RET_VAL(weight_addr_);
    MS_ERROR_IF_NULL_WO_RET_VAL(data_size_addr_);
    MS_ERROR_IF_NULL_WO_RET_VAL(weight_addr_->addr);
    MS_ERROR_IF_NULL_WO_RET_VAL(data_size_addr_->addr);
    int ret = memset_s(weight_addr_->addr, weight_addr_->size, 0x00, weight_addr_->size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memset_s error, errorno(" << ret << ")";
      return;
    }
    ret = memset_s(data_size_addr_->addr, data_size_addr_->size, 0x00, data_size_addr_->size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memset_s error, errorno(" << ret << ")";
      return;
    }
    return;
  }

  // The address pointer of the inputs.
  AddressPtr weight_addr_;
  AddressPtr data_size_addr_;
  AddressPtr new_weight_addr_;
  AddressPtr new_data_size_addr_;
  // The kernel could be called concurrently so we need lock to ensure threadsafe.
  std::mutex weight_mutex_;
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_FED_AVG_KERNEL_H_
