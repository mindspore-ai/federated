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

#include "server/collective_ops_impl.h"
#include "server/local_meta_store.h"
#include "server/iteration.h"
#include "distributed_cache/server.h"
#include "distributed_cache/instance_context.h"

namespace mindspore {
namespace fl {
namespace server {
namespace {
const char kCollectivePhaseRing[] = "ring";
const char kCollectivePhaseGather[] = "gather";
const char kCollectivePhaseReduce[] = "reduce";
const char kCollectivePhaseBroadcast[] = "broadcast";
}  // namespace

void CollectiveOpsImpl::Initialize(const std::shared_ptr<ServerNode> &server_node) {
  MS_EXCEPTION_IF_NULL(server_node);
  server_node_ = server_node;
  node_id_ = server_node_->node_id();
}

template <typename T>
bool CollectiveOpsImpl::RingAllReduce(const std::string &data_name, const void *sendbuff, void *recvbuff,
                                      size_t count) {
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);

  if (recvbuff != sendbuff) {
    size_t src_size = count * sizeof(T);
    size_t dst_size = count * sizeof(T);
    auto ret = memcpy_s(recvbuff, dst_size, sendbuff, src_size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }
  }
  size_t chunk_size = count / rank_size_;
  size_t remainder_size = count % rank_size_;
  std::vector<size_t> chunk_sizes(rank_size_, chunk_size);
  // The rest of the data should be assigned to each chunk.
  for (size_t i = 0; i < remainder_size; i++) {
    chunk_sizes[i]++;
  }
  // Store offsets to get every data chunk's address.
  std::vector<size_t> chunk_offset;
  for (size_t i = 0; i < rank_size_; i++) {
    size_t ofs =
      std::accumulate(chunk_sizes.begin(), chunk_sizes.begin() + i, static_cast<size_t>(0), std::plus<size_t>());
    chunk_offset.push_back(ofs);
  }

  T *output_buff = reinterpret_cast<T *>(recvbuff);
  uint32_t send_to_rank = (rank_id_ + 1) % rank_size_;
  uint32_t recv_from_rank = (rank_id_ - 1 + rank_size_) % rank_size_;
  MS_LOG(DEBUG) << "AllReduce count:" << count << ", rank_size_:" << rank_size_ << ", rank_id_:" << rank_id_
                << ", chunk_size:" << chunk_size << ", remainder_size:" << remainder_size
                << ", chunk_sizes:" << chunk_sizes << ", send_to_rank:" << send_to_rank
                << ", recv_from_rank:" << recv_from_rank;

  return RunRingAllReduce<T>(data_name, send_to_rank, recv_from_rank, chunk_sizes, chunk_offset, output_buff);
}

// Implementation of RingAllReduce.
template <typename T>
bool CollectiveOpsImpl::RunRingAllReduce(const std::string &data_name, uint32_t send_to_rank, uint32_t recv_from_rank,
                                         const std::vector<size_t> &chunk_sizes,
                                         const std::vector<size_t> &chunk_offset, T *output_buff) {
  MS_ERROR_IF_NULL_W_RET_VAL(server_node_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(output_buff, false);
  auto curr_iteration_num = cache::InstanceContext::Instance().iteration_num();

  const auto &send_to_node = server_nodes_[send_to_rank];
  const auto &recv_from_node = server_nodes_[recv_from_rank];

  CollectiveMessageMeta send_meta;
  send_meta.set_enable_flag(true);
  send_meta.set_send_node(node_id_);
  send_meta.set_recv_node(send_to_node.first);
  send_meta.set_iteration(curr_iteration_num);
  send_meta.set_weight_name(data_name);

  CollectiveMessageMeta recv_meta;
  recv_meta.set_enable_flag(true);
  recv_meta.set_send_node(recv_from_node.first);
  recv_meta.set_recv_node(node_id_);
  recv_meta.set_iteration(curr_iteration_num);
  recv_meta.set_weight_name(data_name);

  // Ring ReduceScatter.
  MS_LOG(DEBUG) << "Start Ring ReduceScatter.";
  send_meta.set_phase(kCollectivePhaseRing);
  recv_meta.set_phase(kCollectivePhaseRing);

  const auto &send_address = send_to_node.second;
  for (size_t i = 0; i < rank_size_ - 1; i++) {
    // Step 1: Async send data to next rank.
    size_t send_chunk_index = (rank_id_ - i + rank_size_) % rank_size_;
    T *send_chunk = output_buff + chunk_offset[send_chunk_index];
    send_meta.set_chunk_index(send_chunk_index);
    send_meta.set_for_index(i);
    auto send_chunk_count = chunk_sizes[send_chunk_index];
    auto send_req_id =
      server_node_->CollectiveSendAsync(send_address, send_meta, send_chunk, send_chunk_count * sizeof(T));

    // Step 2: Async receive data to next rank and wait until it's done.
    size_t recv_chunk_index = (rank_id_ - i - 1 + rank_size_) % rank_size_;
    recv_meta.set_chunk_index(recv_chunk_index);
    recv_meta.set_for_index(i);
    T *recv_chunk = output_buff + chunk_offset[recv_chunk_index];
    auto recv_chunk_count = chunk_sizes[recv_chunk_index];
    MS_LOG(DEBUG) << "Ring ReduceScatter send_to_rank:" << send_to_node.first
                  << ", recv_from_rank:" << recv_from_node.first << ", send chunk index:" << send_chunk_index
                  << ", send count:" << send_chunk_count << ", recv chunk index:" << recv_chunk_index
                  << ", recv count:" << recv_chunk_count << ", for index:" << i;

    VectorPtr recv_str;
    auto expect_size = recv_chunk_count * sizeof(T);
    if (!server_node_->CollectiveRecvWait(recv_meta, expect_size, &recv_str, kCollectiveCommTimeout)) {
      MS_LOG(ERROR) << "CollectiveRecvWait failed, send rank id: " << recv_meta.send_node();
      return false;
    }
    auto tmp_recv_chunk = reinterpret_cast<T *>(recv_str->data());
    // Step 3: Reduce the data so we can overlap the time cost of send.
    for (size_t j = 0; j < recv_chunk_count; j++) {
      recv_chunk[j] += tmp_recv_chunk[j];
    }
    // Step 4: Wait until send is done.
    if (!server_node_->Wait(send_req_id, kCollectiveCommTimeout)) {
      MS_LOG(ERROR) << "Wait response of rank " << send_req_id << " failed.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Ring ReduceScatter.";

  // Ring AllGather.
  MS_LOG(DEBUG) << "Start Ring AllGather.";
  send_meta.set_phase(kCollectivePhaseGather);
  recv_meta.set_phase(kCollectivePhaseGather);
  for (size_t i = 0; i < rank_size_ - 1; i++) {
    size_t send_chunk_index = (rank_id_ - i + 1 + rank_size_) % rank_size_;
    T *send_chunk = output_buff + chunk_offset[send_chunk_index];
    send_meta.set_chunk_index(send_chunk_index);
    send_meta.set_for_index(i);
    auto send_chunk_count = chunk_sizes[send_chunk_index];
    auto send_req_id =
      server_node_->CollectiveSendAsync(send_address, send_meta, send_chunk, send_chunk_count * sizeof(T));

    size_t recv_chunk_index = (rank_id_ - i + rank_size_) % rank_size_;
    T *recv_chunk = output_buff + chunk_offset[recv_chunk_index];
    recv_meta.set_chunk_index(recv_chunk_index);
    recv_meta.set_for_index(i);
    auto recv_chunk_count = chunk_sizes[recv_chunk_index];
    MS_LOG(DEBUG) << "Ring AllGather send_to_rank:" << send_to_node.first << ", recv_from_rank:" << recv_from_node.first
                  << ", send chunk index:" << send_chunk_index << ", send count:" << send_chunk_count
                  << ", recv chunk index:" << recv_chunk_index << ", recv count:" << recv_chunk_count
                  << ", for index:" << i;

    VectorPtr recv_str;
    auto expect_size = recv_chunk_count * sizeof(T);
    if (!server_node_->CollectiveRecvWait(recv_meta, expect_size, &recv_str, kCollectiveCommTimeout)) {
      MS_LOG(ERROR) << "CollectiveRecvWait failed, send rank id: " << recv_meta.send_node();
      return false;
    }
    auto ret = memcpy_s(recv_chunk, expect_size, recv_str->data(), recv_str->size());
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")"
                    << ", dest size is " << recv_chunk_count * sizeof(T) << ", src size is " << recv_str->size();
      return false;
    }
    if (!server_node_->Wait(send_req_id, kCollectiveCommTimeout)) {
      MS_LOG(ERROR) << "Wait response of rank " << send_req_id << " failed.";
      return false;
    }
  }
  MS_LOG(DEBUG) << "End Ring AllGather.";
  return true;
}

template <typename T>
bool CollectiveOpsImpl::ReduceBroadcastAllReduce(const std::string &data_name, const void *sendbuff, void *recvbuff,
                                                 size_t count) {
  MS_ERROR_IF_NULL_W_RET_VAL(server_node_, false);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);
  MS_LOG(DEBUG) << "Reduce Broadcast AllReduce rank_size:" << rank_size_ << ", rank_id:" << rank_id_
                << ", node_id:" << node_id_ << ", count:" << count;

  size_t src_size = count * sizeof(T);
  size_t dst_size = count * sizeof(T);
  int ret = memcpy_s(recvbuff, dst_size, sendbuff, src_size);
  if (ret != 0) {
    MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")"
                  << ", dest size is " << dst_size << ", src size is " << src_size;
    return false;
  }
  T *output_buff = reinterpret_cast<T *>(recvbuff);
  // Reduce data to rank 0 process.
  auto curr_iteration_num = cache::InstanceContext::Instance().iteration_num();
  CollectiveMessageMeta send_meta;
  send_meta.set_enable_flag(true);
  send_meta.set_send_node(node_id_);
  send_meta.set_iteration(curr_iteration_num);
  send_meta.set_weight_name(data_name);
  send_meta.set_chunk_index(0);
  send_meta.set_for_index(0);

  CollectiveMessageMeta recv_meta;
  recv_meta.set_enable_flag(true);
  recv_meta.set_recv_node(node_id_);
  recv_meta.set_iteration(curr_iteration_num);
  recv_meta.set_weight_name(data_name);
  recv_meta.set_chunk_index(0);
  recv_meta.set_for_index(0);

  if (rank_id_ == 0) {
    MS_LOG(DEBUG) << "Start Reduce to rank 0 process.";
    recv_meta.set_phase(kCollectivePhaseReduce);
    for (uint32_t i = 1; i < rank_size_; i++) {
      VectorPtr recv_str;
      MS_LOG(DEBUG) << "Reduce rank 0 receive from rank " << i;
      auto &send_node = server_nodes_[i];
      recv_meta.set_send_node(send_node.first);
      auto expect_size = count * sizeof(T);
      if (!server_node_->CollectiveRecvWait(recv_meta, expect_size, &recv_str, kCollectiveCommTimeout)) {
        MS_LOG(ERROR) << "CollectiveRecvWait failed, send rank id: " << recv_meta.send_node();
        return false;
      }
      auto tmp_recv_chunk = reinterpret_cast<T *>(recv_str->data());  // recv_str size has checked in CollectiveWait
      for (size_t j = 0; j < count; j++) {
        output_buff[j] += tmp_recv_chunk[j];
      }
    }
    MS_LOG(DEBUG) << "End Reduce.";
    MS_LOG(DEBUG) << "Start broadcast from rank 0 to other processes.";
    send_meta.set_phase(kCollectivePhaseBroadcast);
    for (uint32_t i = 1; i < rank_size_; i++) {
      MS_LOG(DEBUG) << "Broadcast data to rank " << i;
      auto &recv_node = server_nodes_[i];
      send_meta.set_recv_node(recv_node.first);
      auto send_req_id2 =
        server_node_->CollectiveSendAsync(recv_node.second, send_meta, output_buff, count * sizeof(T));
      if (!server_node_->Wait(send_req_id2, kCollectiveCommTimeout)) {
        MS_LOG(ERROR) << "Wait response of rank " << send_req_id2 << " failed.";
        return false;
      }
    }
    MS_LOG(DEBUG) << "End broadcast.";
  } else {
    MS_LOG(DEBUG) << "Reduce send data to rank 0 process.";
    send_meta.set_phase(kCollectivePhaseReduce);
    auto &rank0_node = server_nodes_[0];
    send_meta.set_recv_node(rank0_node.first);
    auto send_req_id1 = server_node_->CollectiveSendAsync(rank0_node.second, send_meta, sendbuff, count * sizeof(T));
    if (!server_node_->Wait(send_req_id1, kCollectiveCommTimeout)) {
      MS_LOG(ERROR) << "Wait response of rank " << send_req_id1 << " failed.";
      return false;
    }
    MS_LOG(DEBUG) << "End Reduce.";
    MS_LOG(DEBUG) << "Broadcast receive from rank 0.";
    recv_meta.set_phase(kCollectivePhaseBroadcast);
    recv_meta.set_send_node(rank0_node.first);
    VectorPtr recv_str;
    auto expect_size = count * sizeof(T);
    if (!server_node_->CollectiveRecvWait(recv_meta, expect_size, &recv_str, kCollectiveCommTimeout)) {
      MS_LOG(ERROR) << "CollectiveRecvWait failed, send rank id: " << recv_meta.send_node();
      return false;
    }
    ret = memcpy_s(output_buff, expect_size, recv_str->data(), recv_str->size());
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")"
                    << ", dest size is " << expect_size << ", src size is " << recv_str->size();
      return false;
    }
    MS_LOG(DEBUG) << "End broadcast.";
  }
  return true;
}

template <typename T>
bool CollectiveOpsImpl::AllReduce(const std::string &data_name, void *sendbuff, void *recvbuff, size_t count,
                                  const std::map<std::string, std::string> &server_map) {
  // The collective communication API does not support calling Send and Recv concurrently with multiple threads;
  std::unique_lock<std::mutex> lock(mtx_);
  MS_ERROR_IF_NULL_W_RET_VAL(recvbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(sendbuff, false);
  MS_ERROR_IF_NULL_W_RET_VAL(server_node_, false);

  rank_size_ = server_map.size();
  rank_id_ = 0;
  for (auto &item : server_map) {
    if (item.first == node_id_) {
      break;
    }
    rank_id_ += 1;
  }
  if (rank_id_ == server_map.size()) {
    MS_LOG(ERROR) << "Cannot find server " << node_id_ << " in current active server";
    return false;
  }
  if (rank_size_ == 0) {
    MS_LOG(ERROR) << "Rank size should not be 0.";
    return false;
  }
  if (rank_size_ == 1) {
    return true;
  }
  server_nodes_.clear();
  std::transform(server_map.begin(), server_map.end(), std::back_inserter(server_nodes_),
                 [](const std::pair<std::string, std::string> &item) { return item; });

  auto iteration_num = cache::InstanceContext::Instance().iteration_num();
  if (cache::InstanceContext::Instance().HasIterationFailed(iteration_num)) {
    MS_LOG(WARNING) << "Detect iteration " << iteration_num << " has failed";
    return false;
  }
  if (count >= rank_size_) {
    return RingAllReduce<T>(data_name, sendbuff, recvbuff, count);
  } else {
    return ReduceBroadcastAllReduce<T>(data_name, sendbuff, recvbuff, count);
  }
}

template bool CollectiveOpsImpl::AllReduce<float>(const std::string &data_name, void *sendbuff, void *recvbuff,
                                                  size_t count, const std::map<std::string, std::string> &server_map);
template bool CollectiveOpsImpl::AllReduce<size_t>(const std::string &data_name, void *sendbuff, void *recvbuff,
                                                   size_t count, const std::map<std::string, std::string> &server_map);
template bool CollectiveOpsImpl::AllReduce<int>(const std::string &data_name, void *sendbuff, void *recvbuff,
                                                size_t count, const std::map<std::string, std::string> &server_map);

}  // namespace server
}  // namespace fl
}  // namespace mindspore
