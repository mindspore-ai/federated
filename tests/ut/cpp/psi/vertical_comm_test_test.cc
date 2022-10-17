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
#include "vertical/vfl_context.h"
#include "vertical/vertical_server.h"

namespace mindspore {
namespace fl {
class TestVerticalCommunicator : public testing::Test {
 public:
  static void LaunchHttpServer(const std::string &http_server_address, const std::string &http_server_name,
                               std::map<std::string, std::string> remote_server_address) {
    VFLContext::instance()->set_http_server_address(http_server_address);
    VFLContext::instance()->set_http_server_name(http_server_name);
    VFLContext::instance()->set_remote_server_address(remote_server_address);

    WorkerConfigItemPy worker_config_item_py;

    worker_config_item_py.set_primary_key("111");
    worker_config_item_py.set_bucket_num(222);
    worker_config_item_py.set_shard_num(333);
    worker_config_item_py.set_join_type("444");
    VFLContext::instance()->set_worker_config(worker_config_item_py);

    auto &verticalServer = VerticalServer::GetInstance();
    EXPECT_TRUE(verticalServer.StartVerticalCommunicator());
  }

  static void TestDataJoinMsg(const std::string &target_server_name) {
    WorkerConfigItemPy worker_config_item_py = VFLContext::instance()->worker_config();
    auto &verticalServer = VerticalServer::GetInstance();
    WorkerRegisterItemPy workerRegisterItemPy;
    workerRegisterItemPy.set_worker_name("worker1");
    auto workerConfigItemPyResp = verticalServer.Send(target_server_name, workerRegisterItemPy);
    EXPECT_TRUE(verticalServer.DataJoinWaitForStart());

    EXPECT_TRUE(worker_config_item_py.primary_key() == workerConfigItemPyResp.primary_key());
    EXPECT_TRUE(worker_config_item_py.bucket_num() == workerConfigItemPyResp.bucket_num());
    EXPECT_TRUE(worker_config_item_py.shard_num() == workerConfigItemPyResp.shard_num());
    EXPECT_TRUE(worker_config_item_py.join_type() == workerConfigItemPyResp.join_type());
  }

  static void TestAliceCheckMsg(const std::string &target_server_name) {
    auto &verticalServer = VerticalServer::GetInstance();
    psi::AliceCheck aliceCheck;
    aliceCheck.set_bin_id(10);
    aliceCheck.set_wrong_num(20);
    std::vector<std::string> wrong_id = {"123"};
    aliceCheck.set_wrong_id(wrong_id);
    EXPECT_TRUE(verticalServer.Send(target_server_name, aliceCheck));
    psi::AliceCheck aliceCheckResp;
    verticalServer.Receive(target_server_name, &aliceCheckResp);

    EXPECT_TRUE(aliceCheck.bin_id() == aliceCheckResp.bin_id());
    EXPECT_TRUE(aliceCheck.wrong_num() == aliceCheckResp.wrong_num());
    EXPECT_TRUE(aliceCheck.wrong_id()[0] == aliceCheckResp.wrong_id()[0]);
  }

  static void TestAlicePbaAndBFMsg(const std::string &target_server_name) {
    auto &verticalServer = VerticalServer::GetInstance();
    psi::AlicePbaAndBF alicePbaAndBF;
    alicePbaAndBF.set_bin_id(10);
    std::vector<std::string> p_b_a_vct = {"123"};
    alicePbaAndBF.set_p_b_a_vct(p_b_a_vct);
    alicePbaAndBF.set_bf_alice("20");

    EXPECT_TRUE(verticalServer.Send(target_server_name, alicePbaAndBF));
    psi::AlicePbaAndBF alicePbaAndBFResp;
    verticalServer.Receive(target_server_name, &alicePbaAndBFResp);

    EXPECT_TRUE(alicePbaAndBF.bin_id() == alicePbaAndBFResp.bin_id());
    EXPECT_TRUE(alicePbaAndBF.p_b_a_vct()[0] == alicePbaAndBFResp.p_b_a_vct()[0]);
    EXPECT_TRUE(alicePbaAndBF.bf_alice() == alicePbaAndBFResp.bf_alice());
  }

  static void TestBobAlignResultCommMsg(const std::string &target_server_name) {
    auto &verticalServer = VerticalServer::GetInstance();
    psi::BobAlignResult bobAlignResult;
    bobAlignResult.set_bin_id(10);
    std::vector<std::string> align_result = {"123"};
    bobAlignResult.set_align_result(align_result);

    EXPECT_TRUE(verticalServer.Send(target_server_name, bobAlignResult));
    psi::BobAlignResult bobAlignResultResp;
    verticalServer.Receive(target_server_name, &bobAlignResultResp);

    EXPECT_TRUE(bobAlignResult.bin_id() == bobAlignResultResp.bin_id());
    EXPECT_TRUE(bobAlignResult.align_result()[0] == bobAlignResultResp.align_result()[0]);
  }
};

/// Feature: Vertical communicator.
/// Description: Test Vertical Communicator message success for send and receive.
/// Expectation: Get the correct result
TEST_F(TestVerticalCommunicator, TestVerticalCommMsgSuccess) {
  std::string http_server_address = "127.0.0.1:5123";
  std::string http_server_name = "server1";
  std::map<std::string, std::string> remote_server_address = {{http_server_name, http_server_address}};
  LaunchHttpServer(http_server_address, http_server_name, remote_server_address);
  TestDataJoinMsg(http_server_name);
  TestAliceCheckMsg(http_server_name);
  TestAlicePbaAndBFMsg(http_server_name);
  TestBobAlignResultCommMsg(http_server_name);
}
}  // namespace fl
}  // namespace mindspore
