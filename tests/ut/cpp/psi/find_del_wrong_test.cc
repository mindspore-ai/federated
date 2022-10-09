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
#include <random>
#include "gtest/gtest.h"
#include "armour/secure_protocol/psi.h"

namespace mindspore {
namespace fl {
namespace psi {
class TestFindDelWrong : public testing::Test {
 public:
  std::vector<std::string> alice_input;
  std::vector<std::string> align_result;
  std::vector<std::string> wrong_vct;
  std::vector<std::string> fix_vct;
  std::vector<std::string> true_wrong_vct;
  std::vector<std::string> true_align_result;
  PsiCtx psi_ctx_alice;

  void GenData(size_t alice_num, size_t intersect_num, size_t wrong_num) {
    alice_input.resize(alice_num);
    align_result.resize(intersect_num + wrong_num);
    wrong_vct.clear();
    fix_vct.clear();
    true_wrong_vct.clear();
    true_align_result.clear();
    psi_ctx_alice.thread_num = 0;
    // string padding to max_digit_ to have a same length
    size_t max_num = alice_num + wrong_num;
    size_t max_digit = 0;
    while (max_num) {
      max_digit++;
      max_num /= 10;
    }

    for (size_t i = 0; i < alice_num; i++) {
      std::string tmp_str = std::to_string(i);
      alice_input[i] = std::string(max_digit - tmp_str.size(), '0') + tmp_str;
    }
    psi_ctx_alice.input_vct = alice_input;
    psi_ctx_alice.self_num = psi_ctx_alice.input_vct.size();

    std::shuffle(alice_input.begin(), alice_input.end(), std::mt19937(std::random_device()()));
    for (size_t i = 0, start_str = alice_num - intersect_num; i < intersect_num + wrong_num; i++) {
      std::string tmp_str = std::to_string(i + start_str);
      align_result[i] = std::string(max_digit - tmp_str.size(), '0') + tmp_str;
      if (i >= intersect_num)
        true_wrong_vct.emplace_back(align_result[i]);
      else
        true_align_result.emplace_back(align_result[i]);
    }
  }
};

/// Feature: Find false positives in PSI align result.
/// Description: Find items in align result that beyond alice's input.
/// Expectation: Get correct false positives and truly align result.
TEST_F(TestFindDelWrong, PSI_FindWrong) {
  GenData(100, 50, 2);
  FindWrong(psi_ctx_alice, align_result, &wrong_vct, &fix_vct);
  ASSERT_EQ(wrong_vct, true_wrong_vct);
  ASSERT_EQ(fix_vct, true_align_result);

  GenData(1000, 1000, 10);
  FindWrong(psi_ctx_alice, align_result, &wrong_vct, &fix_vct);
  ASSERT_EQ(wrong_vct, true_wrong_vct);
  ASSERT_EQ(fix_vct, true_align_result);

  GenData(100, 0, 1);
  FindWrong(psi_ctx_alice, align_result, &wrong_vct, &fix_vct);
  ASSERT_EQ(wrong_vct, true_wrong_vct);
  ASSERT_EQ(fix_vct, true_align_result);
}

/// Feature: Find and delete false positives in PSI align result.
/// Description: Find and delete items in align result that beyond alice's input.
/// Expectation: Get truly align result in Bob's side.
TEST_F(TestFindDelWrong, PSI_FindDelWrong) {
  GenData(100, 50, 2);
  FindWrong(psi_ctx_alice, align_result, &wrong_vct, &fix_vct);
  DelWrong(&align_result, wrong_vct);
  EXPECT_EQ(align_result, true_align_result);

  GenData(1000, 1000, 10);
  FindWrong(psi_ctx_alice, align_result, &wrong_vct, &fix_vct);
  DelWrong(&align_result, wrong_vct);
  EXPECT_EQ(align_result, true_align_result);

  GenData(100, 0, 1);
  FindWrong(psi_ctx_alice, align_result, &wrong_vct, &fix_vct);
  DelWrong(&align_result, wrong_vct);
  EXPECT_EQ(align_result, true_align_result);
}

}  // namespace psi
}  // namespace fl
}  // namespace mindspore
