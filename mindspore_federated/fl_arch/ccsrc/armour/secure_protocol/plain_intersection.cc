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

#include "armour/secure_protocol/plain_intersection.h"

namespace mindspore {
namespace fl {
namespace psi {
std::vector<std::string> PlainIntersection(const std::vector<std::string> &input_vct, const std::string &comm_role,
                                           size_t thread_num, size_t bin_id, const std::string &target_server_name) {
  std::vector<std::string> ret;
  auto &verticalServer = VerticalServer::GetInstance();
  verticalServer.StartVerticalCommunicator();

  MS_LOG(INFO) << "Start to run plain set intersection, init context...";
  PsiCtx psi_ctx;
  psi_ctx.bin_id = bin_id;
  psi_ctx.thread_num = thread_num;
  psi_ctx.input_vct = input_vct;
  psi_ctx.self_num = psi_ctx.input_vct.size();
  MS_LOG(INFO) << comm_role << " start to hash input and truncate to compare_length (12 bytes)...";
  psi_ctx.input_hash_vct = HashInputs(psi_ctx.input_vct, psi_ctx.thread_num, psi_ctx.chunk_size);
  ParallelSync parallel_sync(thread_num);
  parallel_sync.parallel_for(0, psi_ctx.self_num, psi_ctx.chunk_size, [&](size_t beg, size_t end) {
    for (size_t i = beg; i < end; i++) {
      psi_ctx.input_hash_vct[i] = psi_ctx.input_hash_vct[i].substr(0, psi_ctx.compare_length);
    }
  });

  if (comm_role == "server") {
    MS_LOG(INFO) << "-------------------------- 2. server receive clientPlain -----------------------";
    PlainData clientPlain;
    verticalServer.Receive(target_server_name, &clientPlain);
    std::vector<std::string> recv_vct = clientPlain.plain_data_vct();
    ret = Align(&recv_vct, psi_ctx.input_hash_vct, psi_ctx);
    MS_LOG(INFO) << "-------------------------- 3. server send alignResult -----------------------";
    PlainData alignResult(psi_ctx.bin_id, ret, "alignResult");
    verticalServer.Send(target_server_name, alignResult);
    return ret;
  }

  if (comm_role == "client") {
    MS_LOG(INFO) << "-------------------------- 1. client send clientPlain -----------------------";
    PlainData clientPlain(psi_ctx.bin_id, psi_ctx.input_hash_vct, "clientPlain");
    verticalServer.Send(target_server_name, clientPlain);
    MS_LOG(INFO) << "-------------------------- 4. client receive alignResult -----------------------";
    PlainData alignResult;
    verticalServer.Receive(target_server_name, &alignResult);
    return alignResult.plain_data_vct();
  }

  MS_LOG(ERROR) << "Unknown communication role, input role is " << comm_role;
  return ret;
}

}  // namespace psi
}  // namespace fl
}  // namespace mindspore
