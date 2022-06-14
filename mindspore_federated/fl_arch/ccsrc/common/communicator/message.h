/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_COMMUNICATOR_MESSAGE_H_
#define MINDSPORE_CCSRC_FL_COMMUNICATOR_MESSAGE_H_

#include <string>
#include <memory>

namespace mindspore {
namespace fl {
enum class Protos : uint32_t { RAW = 0, PROTOBUF = 1, FLATBUFFERS = 2 };

struct MessageHeader {
  Protos message_proto_ = Protos::RAW;
  uint32_t message_meta_length_ = 0;
  uint64_t message_length_ = 0;
};
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_COMMUNICATOR_MESSAGE_H_
