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
#include "distributed_cache/summary.h"
#include "distributed_cache/instance_context.h"
#include "distributed_cache/distributed_cache.h"
#include "distributed_cache/redis_keys.h"
#include "distributed_cache/server.h"
#include "common/common.h"

namespace mindspore {
namespace fl {
namespace cache {
namespace {
const char *kSummaryFinishFlag = "Finish";
}

CacheStatus Summary::SubmitSummary(const std::string &summary_pb) {
  auto node_id = Server::Instance().node_id();
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return kCacheNetErr;
  }
  auto key = cache::RedisKeys::GetInstance().IterationSummaryHash();
  constexpr int summary_expire_time_in_seconds = 30;  // 30s
  auto ret = client->HSet(key, node_id, summary_pb);
  (void)client->Expire(key, summary_expire_time_in_seconds);
  return ret;
}

void Summary::GetAllSummaries(std::vector<std::string> *summary_pbs) {
  if (summary_pbs == nullptr) {
    return;
  }
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return;
  }
  auto server_map = Server::Instance().GetAllServers();
  constexpr int max_retry_times = 20;
  auto key = cache::RedisKeys::GetInstance().IterationSummaryHash();
  for (int cur_retry_times = 0; cur_retry_times < max_retry_times; cur_retry_times++) {
    std::unordered_map<std::string, std::string> items;
    auto status = client->HGetAll(key, &items);
    if (status.IsSuccess() && !items.empty()) {
      summary_pbs->clear();
      size_t match_count = 0;
      for (auto &item : items) {
        const auto &node_id = item.first;
        if (server_map.count(node_id)) {
          match_count++;
        }
        summary_pbs->push_back(item.second);
      }
      if (match_count >= server_map.size()) {
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

bool Summary::TryLockSummary(bool *has_finished, bool *has_locked) {
  if (!has_finished || !has_locked) {
    return false;
  }
  *has_finished = false;
  *has_locked = false;
  MS_LOG_INFO << "Begin to lock summary lock";
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return false;
  }
  auto node_id = Server::Instance().node_id();
  auto key = RedisKeys::GetInstance().IterationSummaryLockString();
  constexpr int expire_time_in_seconds = 10;  // lock for 10 seconds
  auto status = client->SetExNx(key, node_id, expire_time_in_seconds);
  if (status == kCacheExist) {
    *has_locked = true;
    std::string val;
    status = client->Get(key, &val);
    if (!status.IsSuccess()) {
      return false;
    }
    *has_finished = (val == kSummaryFinishFlag);
    return false;
  }
  if (!status.IsSuccess()) {
    return false;
  }
  MS_LOG_INFO << "Acquire summary lock successfully";
  return true;
}

void Summary::GetSummaryLockInfo(bool *has_finished, bool *lock_expired) {
  if (!has_finished || !lock_expired) {
    return;
  }
  *lock_expired = false;
  *has_finished = false;
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return;
  }
  auto key = RedisKeys::GetInstance().IterationSummaryLockString();
  std::string val;
  auto status = client->Get(key, &val);
  if (!status.IsSuccess()) {
    if (status == kCacheNil) {
      *lock_expired = true;
    }
    return;
  }
  *has_finished = (val == kSummaryFinishFlag);
}

void Summary::UnlockSummary() {
  auto client = DistributedCacheLoader::Instance().GetOneClient();
  if (client == nullptr) {
    MS_LOG_WARNING << "Get redis client failed";
    return;
  }
  auto node_id = Server::Instance().node_id();
  auto key = RedisKeys::GetInstance().IterationSummaryLockString();
  constexpr int expire_time_in_seconds = 30;  // finish flag for 30 seconds
  auto status = client->SetEx(key, kSummaryFinishFlag, expire_time_in_seconds);
  if (!status.IsSuccess()) {
    MS_LOG_WARNING << "Failed to release summary lock";
    return;
  }
  MS_LOG_INFO << "Release summary lock successfully";
}
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
