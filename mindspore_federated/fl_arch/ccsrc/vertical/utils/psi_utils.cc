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

#include "vertical/utils/psi_utils.h"
#include <vector>
#include <string>

namespace mindspore {
namespace fl {
void CreateAliceCheckProto(datajoin::AliceCheckProto *alice_check_proto, const psi::AliceCheck &alice_check) {
  alice_check_proto->set_bin_id(alice_check.bin_id());
  alice_check_proto->set_wrong_num(alice_check.wrong_num());

  auto wrong_id = alice_check.wrong_id();
  for (const auto &item : wrong_id) {
    alice_check_proto->add_wrong_id(item);
  }

  alice_check_proto->set_msg(alice_check.msg());
}

void CreateClientPSIInitProto(datajoin::ClientPSIInitProto *client_init_proto, const psi::ClientPSIInit &client_init) {
  client_init_proto->set_bin_id(client_init.bin_id());
  client_init_proto->set_psi_type(client_init.psi_type());
  client_init_proto->set_self_size(client_init.self_size());
}

void CreateServerPSIInitProto(datajoin::ServerPSIInitProto *server_init_proto, const psi::ServerPSIInit &server_init) {
  server_init_proto->set_bin_id(server_init.bin_id());
  server_init_proto->set_self_size(server_init.self_size());
  server_init_proto->set_self_role(server_init.self_role());
}

void CreateBobPbProto(datajoin::BobPbProto *bob_p_b_proto, const psi::BobPb &bob_p_b) {
  bob_p_b_proto->set_bin_id(bob_p_b.bin_id());

  auto p_b_vct = bob_p_b.p_b_vct();
  for (const auto &item : p_b_vct) {
    bob_p_b_proto->add_p_b_vct(item);
  }
}

void CreateAlicePbaAndBFProto(datajoin::AlicePbaAndBFProto *alice_pba_bf_proto,
                              const psi::AlicePbaAndBF &alice_pba_bf) {
  alice_pba_bf_proto->set_bin_id(alice_pba_bf.bin_id());

  auto p_b_a_vct = alice_pba_bf.p_b_a_vct();
  for (const auto &item : p_b_a_vct) {
    alice_pba_bf_proto->add_p_b_a_vct(item);
  }

  alice_pba_bf_proto->set_bf_alice(alice_pba_bf.bf_alice());
}

void CreateBobAlignResultProto(datajoin::BobAlignResultProto *bob_align_result_proto,
                               const psi::BobAlignResult &bob_align_result) {
  bob_align_result_proto->set_bin_id(bob_align_result.bin_id());

  auto align_result = bob_align_result.align_result();
  for (const auto &item : align_result) {
    bob_align_result_proto->add_align_result(item);
  }
  bob_align_result_proto->set_msg(bob_align_result.msg());
}

void CreatePlainDataProto(datajoin::PlainDataProto *plain_data_proto, const psi::PlainData &plain_data) {
  plain_data_proto->set_bin_id(plain_data.bin_id());

  auto plain_data_vct = plain_data.plain_data_vct();
  for (const auto &item : plain_data_vct) {
    plain_data_proto->add_plain_data_vct(item);
  }

  plain_data_proto->set_msg(plain_data.msg());
}

psi::BobPb ParseBobPbProto(const datajoin::BobPbProto &bobPbProto) {
  psi::BobPb bobPb;
  bobPb.set_bin_id(bobPbProto.bin_id());
  std::vector<std::string> p_b_vct;
  int p_b_vct_size = bobPbProto.p_b_vct_size();
  for (int i = 0; i < p_b_vct_size; i++) {
    p_b_vct.push_back(bobPbProto.p_b_vct(i));
  }
  bobPb.set_p_b_vct(p_b_vct);
  MS_LOG(INFO) << "(bob_p_b) bin_id is " << bobPb.bin_id() << ", vector size is " << bobPb.p_b_vct().size();
  return bobPb;
}

psi::ClientPSIInit ParseClientPSIInitProto(const datajoin::ClientPSIInitProto &clientPSIInitProto) {
  psi::ClientPSIInit clientPSIInit;
  clientPSIInit.set_bin_id(clientPSIInitProto.bin_id());
  clientPSIInit.set_psi_type(clientPSIInitProto.psi_type());
  clientPSIInit.set_self_size(clientPSIInitProto.self_size());
  MS_LOG(INFO) << "(client_psi_init) bin_id is " << clientPSIInit.bin_id() << ", psi_type is "
               << clientPSIInit.psi_type() << ", self_size is " << clientPSIInit.self_size();
  return clientPSIInit;
}

psi::ServerPSIInit ParseServerPSIInitProto(const datajoin::ServerPSIInitProto &serverPSIInitProto) {
  psi::ServerPSIInit serverPSIInit;
  serverPSIInit.set_bin_id(serverPSIInitProto.bin_id());
  serverPSIInit.set_self_size(serverPSIInitProto.self_size());
  serverPSIInit.set_self_role(serverPSIInitProto.self_role());
  MS_LOG(INFO) << "(server_psi_init) bin_id is " << serverPSIInit.bin_id() << ", self_size is "
               << serverPSIInit.self_size() << ", self_role is " << serverPSIInit.self_role();
  return serverPSIInit;
}

psi::AlicePbaAndBF ParseAlicePbaAndBFProto(const datajoin::AlicePbaAndBFProto &alicePbaAndBFProto) {
  psi::AlicePbaAndBF alicePbaAndBF;
  alicePbaAndBF.set_bin_id(alicePbaAndBFProto.bin_id());
  std::vector<std::string> p_b_vct;
  int p_b_a_vct_size = alicePbaAndBFProto.p_b_a_vct_size();
  for (int i = 0; i < p_b_a_vct_size; i++) {
    p_b_vct.push_back(alicePbaAndBFProto.p_b_a_vct(i));
  }
  alicePbaAndBF.set_p_b_a_vct(p_b_vct);
  alicePbaAndBF.set_bf_alice(alicePbaAndBFProto.bf_alice());
  MS_LOG(INFO) << "(alice_pba_bf) bin_id is " << alicePbaAndBF.bin_id() << ", alice_p_b_a size is "
               << alicePbaAndBF.p_b_a_vct().size() << ", bf_alice size is " << alicePbaAndBF.bf_alice().size();
  return alicePbaAndBF;
}

psi::BobAlignResult ParseBobAlignResultProto(const datajoin::BobAlignResultProto &bobAlignResultProto) {
  psi::BobAlignResult bobAlignResult;
  bobAlignResult.set_bin_id(bobAlignResultProto.bin_id());
  std::vector<std::string> align_result;
  int align_result_size = bobAlignResultProto.align_result_size();
  for (int i = 0; i < align_result_size; i++) {
    align_result.push_back(bobAlignResultProto.align_result(i));
  }
  bobAlignResult.set_align_result(align_result);
  MS_LOG(INFO) << "(bob_align_result), bin_id is " << bobAlignResult.bin_id() << ", vector size is "
               << bobAlignResult.align_result().size();
  return bobAlignResult;
}

psi::AliceCheck ParseAliceCheckProto(const datajoin::AliceCheckProto &aliceCheckProto) {
  psi::AliceCheck aliceCheck;
  aliceCheck.set_bin_id(aliceCheckProto.bin_id());
  aliceCheck.set_wrong_num(aliceCheckProto.wrong_num());
  std::vector<std::string> wrong_id_vct;
  int wrong_id_size = aliceCheckProto.wrong_id_size();
  for (int i = 0; i < wrong_id_size; i++) {
    wrong_id_vct.push_back(aliceCheckProto.wrong_id(i));
  }
  aliceCheck.set_wrong_id(wrong_id_vct);
  MS_LOG(INFO) << "(alice_check) bin_id is " << aliceCheck.bin_id() << ", wrong_id size is "
               << aliceCheck.wrong_id().size();
  return aliceCheck;
}

psi::PlainData ParsePlainDataProto(const datajoin::PlainDataProto &plainDataProto) {
  psi::PlainData plainData;
  plainData.set_bin_id(plainDataProto.bin_id());
  std::vector<std::string> plain_data_vct;
  int plain_data_vct_size = plainDataProto.plain_data_vct_size();
  for (int i = 0; i < plain_data_vct_size; i++) {
    plain_data_vct.push_back(plainDataProto.plain_data_vct(i));
  }
  plainData.set_plain_data_vct(plain_data_vct);
  plainData.set_msg(plainDataProto.msg());
  MS_LOG(INFO) << "(plain_data) bin_id is " << plainData.bin_id() << ", vector size is "
               << plainData.plain_data_vct().size() << ", message: " << plainData.msg();
  return plainData;
}
}  // namespace fl
}  // namespace mindspore
