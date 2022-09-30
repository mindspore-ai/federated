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

#include <memory>
#include "gtest/gtest.h"
#include "common/parallel_for.h"

namespace mindspore {
namespace fl {
class TestParallelFor : public testing::Test {
 public:
  static size_t ParallelSum(size_t begin_num, size_t end_num, size_t thread_num) {
    size_t sum = 0;
    ParallelSync parallel_sync(thread_num);
    size_t thread_num_ = parallel_sync.get_thread_num();
    std::vector<size_t> thread_sum(thread_num_, 0);

    parallel_sync.parallel_for(0, thread_num_, 0, [&](size_t beg, size_t end) {
      for (size_t i = beg; i < end; i++) {
        for (size_t j = begin_num + i; j <= end_num; j += thread_num_) {
          thread_sum[i] += j;
        }
      }
    });
    for (size_t i = 0; i < thread_num_; i++) sum += thread_sum[i];
    return sum;
  }

  std::vector<size_t> ParallelAddItem(size_t item_num, size_t thread_num) {
    std::atomic<size_t> idx(0);
    std::vector<size_t> ret(item_num);
    ParallelSync parallel_sync(thread_num);
    parallel_sync.parallel_for(0, item_num, 0, [&](size_t beg, size_t end) {
      for (size_t i = beg; i < end; i++) {
        ret[idx++] = i;
      }
    });
    sort(ret.begin(), ret.end());

    return ret;
  }
};

/// Feature: Parallel integer summation.
/// Description: Test basic for-loop integer summation.
/// Expectation: Get correct sum result.
TEST_F(TestParallelFor, SumTest) {
  EXPECT_EQ(ParallelSum(1, 100, 0), 5050);
  EXPECT_EQ(ParallelSum(1, 100, 10), 5050);
  EXPECT_EQ(ParallelSum(1, 100, 17), 5050);
  EXPECT_EQ(ParallelSum(0, 10, 0), 55);
  EXPECT_EQ(ParallelSum(0, 10, 50), 55);
}

/// Feature: Push back items into vector in parallel.
/// Description: Test mutex writing.
/// Expectation: Get a vector with correct items.
TEST_F(TestParallelFor, AddItemTest) {
  std::vector<size_t> vec1(1000);
  for (size_t i = 0; i < 1000; i++) vec1[i] = i;
  EXPECT_TRUE(ParallelAddItem(1000, 0) == vec1);
  EXPECT_TRUE(ParallelAddItem(1000, 13) == vec1);

  std::vector<size_t> vec2(10);
  for (size_t i = 0; i < 10; i++) vec2[i] = i;
  EXPECT_TRUE(ParallelAddItem(10, 0) == vec2);
}

}  // namespace fl
}  // namespace mindspore
