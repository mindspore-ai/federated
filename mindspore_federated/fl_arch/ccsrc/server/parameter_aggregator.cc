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

#include "server/parameter_aggregator.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

namespace mindspore {
namespace fl {
namespace server {
bool ParameterAggregator::Init(const std::string &param_name, size_t threshold_count) {
  memory_register_ = std::make_shared<MemoryRegister>();

  required_push_count_ = threshold_count;
  // The required_pull_count_ is the count for Pull, which should be the same as required_push_count_.
  // required_pull_count_ normally used in parameter server training mode.
  required_pull_count_ = threshold_count;

  MS_LOG(INFO) << "Start initializing kernels for " << param_name;
  if (!InitAggregationKernels(param_name)) {
    MS_LOG(EXCEPTION) << "Initializing aggregation kernels failed.";
    return false;
  }
  return true;
}

bool ParameterAggregator::ReInitForScaling() {
  auto result = std::find_if(aggregation_kernel_parameters_.begin(), aggregation_kernel_parameters_.end(),
                             [](auto aggregation_kernel) {
                               MS_ERROR_IF_NULL_W_RET_VAL(aggregation_kernel.first, true);
                               return !aggregation_kernel.first->ReInitForScaling();
                             });
  if (result != aggregation_kernel_parameters_.end()) {
    MS_LOG(ERROR) << "Reinitializing aggregation kernel after scaling failed";
    return false;
  }
  return true;
}

bool ParameterAggregator::ReInitForUpdatingHyperParams(size_t aggr_threshold) {
  required_push_count_ = aggr_threshold;
  required_pull_count_ = aggr_threshold;
  auto result = std::find_if(aggregation_kernel_parameters_.begin(), aggregation_kernel_parameters_.end(),
                             [aggr_threshold](auto aggregation_kernel) {
                               MS_ERROR_IF_NULL_W_RET_VAL(aggregation_kernel.first, true);
                               return !aggregation_kernel.first->ReInitForUpdatingHyperParams(aggr_threshold);
                             });
  if (result != aggregation_kernel_parameters_.end()) {
    MS_LOG(ERROR) << "Reinitializing aggregation kernel after scaling failed";
    return false;
  }
  return true;
}

bool ParameterAggregator::UpdateData(const std::map<std::string, Address> &new_data) {
  std::map<std::string, AddressPtr> &name_to_addr = memory_register_->addresses();
  for (const auto &data : new_data) {
    const std::string &name = data.first;
    if (name_to_addr.count(name) == 0) {
      continue;
    }

    MS_ERROR_IF_NULL_W_RET_VAL(name_to_addr[name], false);
    MS_ERROR_IF_NULL_W_RET_VAL(name_to_addr[name]->addr, false);
    MS_ERROR_IF_NULL_W_RET_VAL(data.second.addr, false);
    MS_LOG(DEBUG) << "Update data for " << name << ". Destination size: " << name_to_addr[name]->size
                  << ". Source size: " << data.second.size;
    int ret = memcpy_s(name_to_addr[name]->addr, name_to_addr[name]->size, data.second.addr, data.second.size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }
  }
  return true;
}

bool ParameterAggregator::LaunchAggregators() {
  for (auto &aggregator_with_params : aggregation_kernel_parameters_) {
    KernelParams &params = aggregator_with_params.second;
    std::shared_ptr<kernel::AggregationKernelMod> aggr_kernel = aggregator_with_params.first;
    MS_ERROR_IF_NULL_W_RET_VAL(aggr_kernel, false);
    bool ret = aggr_kernel->Launch(params.inputs, params.workspace, params.outputs);
    if (!ret) {
      MS_LOG(ERROR) << "Launching aggregation kernel " << typeid(aggr_kernel.get()).name() << " failed.";
      return false;
    }
  }
  return true;
}

AddressPtr ParameterAggregator::GetWeight() {
  if (memory_register_ == nullptr) {
    MS_LOG(ERROR)
      << "The memory register of ParameterAggregator is nullptr. Please initialize ParameterAggregator first.";
    return nullptr;
  }
  std::map<std::string, AddressPtr> &name_to_addr = memory_register_->addresses();
  return name_to_addr["weight"];
}

void ParameterAggregator::ResetAggregationStatus() {
  for (auto &aggregator_with_params : aggregation_kernel_parameters_) {
    std::shared_ptr<kernel::AggregationKernelMod> aggr_kernel = aggregator_with_params.first;
    if (aggr_kernel == nullptr) {
      MS_LOG(ERROR) << "The aggregation kernel is nullptr.";
      continue;
    }
    aggr_kernel->Reset();
  }
  return;
}

void ParameterAggregator::ResetPullingStatus() {
  pulling_done_ = false;
  current_pull_count_ = 0;
}

bool ParameterAggregator::IsAggregationDone() const {
  // Only consider aggregation done after each aggregation kernel is done.
  for (auto &aggregator_with_params : aggregation_kernel_parameters_) {
    std::shared_ptr<kernel::AggregationKernelMod> aggr_kernel = aggregator_with_params.first;
    MS_ERROR_IF_NULL_W_RET_VAL(aggr_kernel, false);
    if (!aggr_kernel->IsAggregationDone()) {
      return false;
    }
  }
  return true;
}

bool ParameterAggregator::RunAggregation() {
  for (auto &aggregator_with_params : aggregation_kernel_parameters_) {
    std::shared_ptr<kernel::AggregationKernelMod> aggr_kernel = aggregator_with_params.first;
    MS_ERROR_IF_NULL_W_RET_VAL(aggr_kernel, false);
    if (!aggr_kernel->AllReduce()) {
      return false;
    }
  }
  return true;
}

bool ParameterAggregator::IsPullingDone() const { return pulling_done_; }

bool ParameterAggregator::requires_aggr() const { return requires_aggr_; }

bool ParameterAggregator::InitAggregationKernels(const std::string &param_name) {
  std::vector<std::string> aggr_kernel_names = SelectAggregationAlgorithm();
  for (const std::string &name : aggr_kernel_names) {
    auto aggr_kernel = kernel::AggregationKernelFactory::GetInstance().Create(name);
    if (aggr_kernel == nullptr) {
      MS_LOG(EXCEPTION) << "Fail to create aggregation kernel " << name ;
      return false;
    }

    // set_done_count must be called before InitKernel because InitKernel may use this count.
    aggr_kernel->set_done_count(required_push_count_);
    aggr_kernel->InitKernel(param_name);

    if (!AssignMemory(param_name, aggr_kernel, memory_register_)) {
      MS_LOG(EXCEPTION) << "Assigning memory for kernel " << name << " failed.";
      return false;
    }

    if (!GenerateAggregationKernelParams(aggr_kernel, memory_register_)) {
      MS_LOG(EXCEPTION) << "Generating aggregation kernel parameters for " << name << " failed.";
      return false;
    }
  }
  return true;
}

bool ParameterAggregator::AssignMemory(const std::string &param_name, std::shared_ptr<kernel::AggregationKernelMod> server_kernel,
                                       const std::shared_ptr<MemoryRegister> &memory_register) {
  MS_EXCEPTION_IF_NULL(server_kernel);
  MS_EXCEPTION_IF_NULL(memory_register);
  const std::vector<size_t> &input_size_list = server_kernel->GetInputSizeList();
  const std::vector<std::string> &input_names = server_kernel->input_names();
  MS_LOG(INFO) << "The input names are " << input_names;
  auto &feature_maps = FLContext::instance()->feature_maps();

  for (size_t i = 0; i < input_names.size(); i++) {
    const std::string &input_name = input_names[i];
    if (memory_register->addresses().count(input_name) != 0) {
      MS_LOG(DEBUG) << "The memory for " << input_name << " is already assigned.";
      continue;
    }
    size_t size = input_size_list[i];
    auto input_addr = std::make_unique<char[]>(size);
    if (input_name == kWeight) {
      const auto& feature_map = feature_maps[param_name];
      const auto& weight_data = feature_map.weight_data;
      float data[weight_data.size()];
      std::copy(weight_data.begin(), weight_data.end(), data);
      int ret = memcpy_s(input_addr.get(), size, data, size);
      if (ret != 0) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return false;
      }
    }
    memory_register->RegisterArray(input_name, &input_addr, size);
    MS_LOG(INFO) << "Assign new memory for " << input_name;
  }
  return true;
}

bool ParameterAggregator::GenerateAggregationKernelParams(
  const std::shared_ptr<kernel::AggregationKernelMod> &aggr_kernel,
  const std::shared_ptr<MemoryRegister> &memory_register) {
  MS_ERROR_IF_NULL_W_RET_VAL(aggr_kernel, false);
  MS_ERROR_IF_NULL_W_RET_VAL(memory_register, false);
  KernelParams aggr_params = {};

  const std::vector<std::string> &input_names = aggr_kernel->input_names();
  (void)std::transform(input_names.begin(), input_names.end(), std::back_inserter(aggr_params.inputs),
                       [&](const std::string &name) { return memory_register->addresses()[name]; });

  const std::vector<std::string> &workspace_names = aggr_kernel->workspace_names();
  (void)std::transform(workspace_names.begin(), workspace_names.end(), std::back_inserter(aggr_params.workspace),
                       [&](const std::string &name) { return memory_register->addresses()[name]; });

  const std::vector<std::string> &output_names = aggr_kernel->output_names();
  (void)std::transform(output_names.begin(), output_names.end(), std::back_inserter(aggr_params.outputs),
                       [&](const std::string &name) { return memory_register->addresses()[name]; });

  aggr_kernel->SetParameterAddress(aggr_params.inputs, aggr_params.workspace, aggr_params.outputs);
  aggregation_kernel_parameters_.push_back(std::make_pair(aggr_kernel, aggr_params));
  return true;
}

std::vector<std::string> ParameterAggregator::SelectAggregationAlgorithm() {
  std::vector<std::string> aggregation_algorithm = {};
  (void)aggregation_algorithm.emplace_back("FedAvg");
  MS_LOG(INFO) << "Aggregation algorithm selection result: " << aggregation_algorithm;
  return aggregation_algorithm;
}

bool ParameterAggregator::JudgeRequiredAggr() {
  return true;
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
