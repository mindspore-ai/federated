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
#include "vertical/python/worker_register_py.h"
#include <functional>
#include <vector>

namespace mindspore {
namespace fl {
void WorkerRegisterItemPy::set_worker_name(const std::string &worker_name) { worker_name_ = worker_name; }
std::string WorkerRegisterItemPy::worker_name() const { return worker_name_; }
}  // namespace fl
}  // namespace mindspore
