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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_KERNEL_FACTORY_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_KERNEL_FACTORY_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include "common/common.h"
#include "server/kernel/params_info.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
// KernelFactory is used to select and build kernels in server.
// Unlike normal MindSpore operator kernels, the server defines multiple types of kernels. For example: Aggregation
// Kernel, Forward Kernel, etc. So we define KernelFactory as a template class for register of all
// types of kernels.

// Create a server aggregation kernel
// Typename K refers to the shared_ptr of the kernel type.
// Typename C refers to the creator function of the kernel.
template <typename K, typename C>
class KernelFactory {
 public:
  KernelFactory() = default;
  virtual ~KernelFactory() = default;

  static KernelFactory &GetInstance() {
    static KernelFactory instance;
    return instance;
  }

  // Kernels are registered by parameter information and its creator(constructor).
  void Register(const std::string &name, const ParamsInfo &params_info, C &&creator) {
    name_to_creator_map_[name].push_back(std::make_pair(params_info, creator));
  }

  K Create(const std::string &name) {
    if (name_to_creator_map_.count(name) == 0) {
      MS_LOG(ERROR) << "Creating kernel failed: " << name << " is not registered.";
    }
    for (const auto &name_type_creator : name_to_creator_map_[name]) {
      const ParamsInfo &params_info = name_type_creator.first;
      const C &creator = name_type_creator.second;
      auto kernel = creator();
      kernel->set_params_info(params_info);
      return kernel;
    }
    return nullptr;
  }

 private:
  KernelFactory(const KernelFactory &) = delete;
  KernelFactory &operator=(const KernelFactory &) = delete;

  // Judge whether the server kernel can be created according to registered ParamsInfo.
  virtual bool Matched(const ParamsInfo &params_info) { return true; }

  // Generally, a server kernel can correspond to several ParamsInfo which is registered by the method 'Register' in
  // server kernel's *.cc files.
  std::unordered_map<std::string, std::vector<std::pair<ParamsInfo, C>>> name_to_creator_map_;
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_KERNEL_FACTORY_H_
