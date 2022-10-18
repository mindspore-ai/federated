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

#include <random>
#include <vector>
#include <algorithm>
#include <string>
#include <memory>
#include "armour/base_crypto/hash.h"
#include "armour/util/io_util.h"
#include "armour/secure_protocol/psi.h"
#include "vertical/vertical_server.h"

namespace mindspore {
namespace fl {
namespace psi {

std::vector<std::string> Align(std::vector<std::string> *alice_vct, const std::vector<std::string> &bob_vct,
                               const PsiCtx &psi_ctx) {
  MS_LOG(INFO) << "Bob start doing align.";
  time_t time_start;
  time_t time_end;
  time(&time_start);

  std::sort(alice_vct->begin(), alice_vct->end());
  time(&time_end);
  MS_LOG(INFO) << "Sort alice's data, time cost: " << difftime(time_end, time_start) << " s.";

  time(&time_start);
  std::vector<std::string> align_results_vector(std::min(alice_vct->size(), bob_vct.size()));
  std::atomic<size_t> idx(0);
  ParallelSync parallel_sync(psi_ctx.thread_num);
  parallel_sync.parallel_for(0, bob_vct.size(), psi_ctx.chunk_size, [&](size_t beg, size_t end) {
    for (size_t i = beg; i < end; i++) {
      if (std::binary_search(alice_vct->begin(), alice_vct->end(), bob_vct[i])) {
        align_results_vector[idx++] = psi_ctx.input_vct[i];
      }
    }
  });
  align_results_vector.resize(idx);
  time(&time_end);
  MS_LOG(INFO) << "Do align, time cost: " << difftime(time_end, time_start) << " s.";
  MS_LOG(INFO) << "PSI num is: " << align_results_vector.size();

  return align_results_vector;
}

std::vector<std::string> Align(const std::vector<std::string> &p_b_a_bI_vct, const BloomFilter &bf,
                               const PsiCtx &psi_ctx) {
  MS_LOG(INFO) << "Bob start doing filter_align";
  time_t time_start;
  time_t time_end;
  time(&time_start);

  std::vector<std::string> align_results_vector(std::min(p_b_a_bI_vct.size(), bf.input_num_));
  std::atomic<size_t> idx(0);
  ParallelSync parallel_sync(psi_ctx.thread_num);
  parallel_sync.parallel_for(0, p_b_a_bI_vct.size(), psi_ctx.chunk_size, [&](size_t beg, size_t end) {
    for (size_t i = beg; i < end; i++) {
      if (bf.LookUp(p_b_a_bI_vct[i])) {
        align_results_vector[idx++] = psi_ctx.input_vct[i];
      }
    }
  });
  align_results_vector.resize(idx);

  time(&time_end);
  MS_LOG(INFO) << "Do filter_align, align_results_vector time cost: " << difftime(time_end, time_start)
               << " s. align_results_vector size is: " << align_results_vector.size();
  return align_results_vector;
}

void FindWrong(const PsiCtx &psi_ctx, const std::vector<std::string> &align_result, std::vector<std::string> *wrong_vct,
               std::vector<std::string> *fix_vct) {
  MS_LOG(INFO) << "alice start doing FindWrong";
  time_t time_start;
  time_t time_end;
  time(&time_start);

  std::vector<unsigned char> flag_vct(align_result.size(), 0);
  ParallelSync parallel_sync(psi_ctx.thread_num);
  parallel_sync.parallel_for(0, psi_ctx.self_num, psi_ctx.chunk_size, [&](size_t beg, size_t end) {
    for (size_t i = beg; i < end; i++) {
      size_t index = distance(align_result.begin(),
                              std::lower_bound(align_result.begin(), align_result.end(), psi_ctx.input_vct[i]));
      if (index < align_result.size() && align_result[index] == psi_ctx.input_vct[i]) {
        flag_vct[index] = 1;
      }
    }
  });
  for (size_t i = 0; i < flag_vct.size(); i++) {
    if (flag_vct[i] == 1) {
      fix_vct->emplace_back(align_result[i]);
    } else {
      wrong_vct->emplace_back(align_result[i]);
    }
  }
  time(&time_end);
  MS_LOG(INFO) << "Do FindWrong, wrong_vct time cost: " << difftime(time_end, time_start) << " s.";
  MS_LOG(INFO) << "Number of error samples found by Alice is " << wrong_vct->size();
}

void DelWrong(std::vector<std::string> *align_results_vector, const std::vector<std::string> &recv_wrong_vct) {
  MS_LOG(INFO) << "Bob start doing DelWrong";
  if (recv_wrong_vct.empty()) {
    MS_LOG(INFO) << "Alice does not find wrong id, return directly .";
    return;
  }
  time_t time_start;
  time_t time_end;
  time(&time_start);
  size_t del_num = 0;
  for (const auto &item : recv_wrong_vct) {
    for (auto iter = align_results_vector->begin(); iter != align_results_vector->end(); iter++) {
      if (*iter == item) {
        align_results_vector->erase(iter);
        del_num++;
        break;
      }
    }
  }
  if (del_num != recv_wrong_vct.size()) {
    MS_LOG(ERROR) << "Bob receives some id that Bob doesn't have.";
  }
  time(&time_end);
  MS_LOG(INFO) << "Do DelWrong, time cost: " << difftime(time_end, time_start) << " s.";
}

std::vector<std::string> RunInverseFilterEcdhPsi(const PsiCtx &psi_ctx_alice, const PsiCtx &psi_ctx_bob) {
  // bob
  MS_LOG(INFO)
    << "  -------------------------- 0.[offline] Bob start hashing and computing p2^b...----------------------";
  std::vector<std::string> bob_input_hash_vct =
    HashInputs(psi_ctx_bob.input_vct, psi_ctx_bob.thread_num, psi_ctx_bob.chunk_size);
  auto p_b_vct =
    psi_ctx_bob.ecc->HashToCurveAndMul(bob_input_hash_vct, psi_ctx_bob.compress_length, psi_ctx_bob.compress_length);
  std::vector<std::string>().swap(bob_input_hash_vct);
  MS_LOG(INFO) << "  -------------------------- 1. bob send bobPb -----------------------";
  BobPb bob_p_b(psi_ctx_bob.bin_id, p_b_vct);
  Send(bob_p_b);

  // alice
  MS_LOG(INFO)
    << "  -------------------------- 0.[offline] Alice start hashing and computing p1^a...----------------------";
  std::vector<std::string> alice_input_hash_vct =
    HashInputs(psi_ctx_alice.input_vct, psi_ctx_alice.thread_num, psi_ctx_alice.chunk_size);
  auto p_a_vct = psi_ctx_alice.ecc->HashToCurveAndMul(alice_input_hash_vct, LENGTH_32, psi_ctx_alice.compare_length);
  std::vector<std::string>().swap(alice_input_hash_vct);
  BloomFilter bf_alice(p_a_vct, psi_ctx_alice.thread_num, psi_ctx_alice.neg_log_fp_rate);
  std::vector<std::string>().swap(p_a_vct);

  MS_LOG(INFO) << "  -------------------------- 2. alice receive bob_p_b -----------------------";
  BobPb bob_p_b_recv;
  Recv(&bob_p_b_recv);
  MS_LOG(INFO) << "Alice start decompress and compute p2^b^a ";
  auto p_b_a_vct =
    psi_ctx_alice.ecc->DcpsAndMul(bob_p_b_recv.p_b_vct(), psi_ctx_alice.compress_length, psi_ctx_alice.compress_length);
  bob_p_b_recv.set_empty();

  MS_LOG(INFO) << "  -------------------------- 3. alice send AlicePbaAndBFProto -----------------------";
  AlicePbaAndBF alice_p_b_a_bf(psi_ctx_alice.bin_id, p_b_a_vct, bf_alice.GetData());
  Send(alice_p_b_a_bf);

  // bob
  MS_LOG(INFO) << "  -------------------------- 4. bob receive alice_p_b_a_bf -----------------------";
  AlicePbaAndBF alice_p_b_a_bf_recv;
  Recv(&alice_p_b_a_bf_recv);
  MS_LOG(INFO) << "Bob start decompress and compute p2^b^a^(b^-1) ";
  auto p_b_a_bI_vct =
    psi_ctx_bob.ecc->DcpsAndInverseMul(alice_p_b_a_bf_recv.p_b_a_vct(), LENGTH_32, psi_ctx_bob.compare_length);

  BloomFilter bf_alice_recv(alice_p_b_a_bf_recv.bf_alice(), psi_ctx_bob.peer_num, psi_ctx_bob.neg_log_fp_rate);
  alice_p_b_a_bf_recv.set_empty();
  auto align_results_vector = Align(p_b_a_bI_vct, bf_alice_recv, psi_ctx_bob);
  std::vector<std::string>().swap(p_b_a_bI_vct);
  MS_LOG(INFO) << "Number of false positive cases: "
               << static_cast<int>(align_results_vector.size() - psi_ctx_bob.input_vct.size() / 2);

  if (!psi_ctx_bob.need_check) {
    MS_LOG(INFO) << "  -------------------------- 5. bob send align_result -----------------------";
    BobAlignResult bob_align_result(psi_ctx_bob.bin_id, align_results_vector);
    Send(bob_align_result);

    MS_LOG(INFO) << "  -------------------------- 6. alice receive align_result -----------------------";
    BobAlignResult bob_align_result_recv;
    Recv(&bob_align_result_recv);
    return bob_align_result_recv.align_result();
  } else {
    time_t time_start;
    time_t time_end;
    time(&time_start);
    std::sort(align_results_vector.begin(), align_results_vector.end());
    time(&time_end);
    MS_LOG(INFO) << "Bob sort align result, time cost: " << difftime(time_end, time_start) << " s.";
    MS_LOG(INFO) << "  -------------------------- 5. bob send align_result -----------------------";
    BobAlignResult bob_align_result(psi_ctx_bob.bin_id, align_results_vector);
    Send(bob_align_result);

    MS_LOG(INFO) << "  -------------------------- 6. alice receive align_result -----------------------";
    std::vector<std::string> wrong_vct;
    std::vector<std::string> fix_vct;
    BobAlignResult bob_align_result_recv;
    Recv(&bob_align_result_recv);
    FindWrong(psi_ctx_alice, bob_align_result_recv.align_result(), &wrong_vct, &fix_vct);
    bob_align_result_recv.set_empty();

    MS_LOG(INFO) << "  -------------------------- 7. alice send wrong_id -----------------------";
    AliceCheck alice_check(psi_ctx_alice.bin_id, wrong_vct.size(), wrong_vct);
    Send(alice_check);

    MS_LOG(INFO) << "  -------------------------- 8. bob receive wrong_id -----------------------";
    AliceCheck alice_check_recv;
    Recv(&alice_check_recv);
    DelWrong(&align_results_vector, alice_check_recv.wrong_id());
    return align_results_vector;
  }
}

std::vector<std::string> RunPSIDemo(const std::vector<std::string> &alice_input,
                                    const std::vector<std::string> &bob_input, size_t thread_num) {
  std::vector<std::string> ret;
  MS_LOG(INFO) << "Start RunEcdhPsi, init config...";
  PsiCtx psi_ctx_alice;
  psi_ctx_alice.thread_num = thread_num;
  psi_ctx_alice.ecc =
    std::make_unique<ECC>(psi_ctx_alice.curve_name, psi_ctx_alice.thread_num, psi_ctx_alice.chunk_size);
  psi_ctx_alice.input_vct = alice_input;
  psi_ctx_alice.self_num = psi_ctx_alice.input_vct.size();
  psi_ctx_alice.peer_num = bob_input.size();
  psi_ctx_alice.compare_length = LENGTH_32;

  if (!psi_ctx_alice.CheckPsiCtxOK()) {
    MS_LOG(ERROR) << "Set PSI CTX ERROR!";
    return ret;
  }
  //  -------------------------- 1. client send clientPsiInit -----------------------
  ClientPSIInit client_psi_init(psi_ctx_alice.bin_id, psi_ctx_alice.psi_type, psi_ctx_alice.self_num);
  Send(client_psi_init);

  //  -------------------------- 2. server receive clientPsiInit -----------------------
  ClientPSIInit client_psi_init_recv;
  Recv(&client_psi_init_recv);

  PsiCtx psi_ctx_bob;
  psi_ctx_bob.thread_num = thread_num;
  psi_ctx_bob.ecc = std::make_unique<ECC>(psi_ctx_bob.curve_name, psi_ctx_bob.thread_num, psi_ctx_bob.chunk_size);
  psi_ctx_bob.input_vct = bob_input;
  psi_ctx_bob.self_num = psi_ctx_bob.input_vct.size();
  psi_ctx_bob.peer_num = alice_input.size();
  psi_ctx_bob.compare_length = LENGTH_32;
  psi_ctx_bob.SetRole(client_psi_init_recv.self_size());

  if (!psi_ctx_bob.CheckPsiCtxOK()) {
    MS_LOG(ERROR) << "Set PSI CTX ERROR!";
    return ret;
  }

  //  -------------------------- 3. server send serverPsiInit -----------------------
  ServerPSIInit server_psi_init(psi_ctx_bob.bin_id, psi_ctx_bob.self_num, psi_ctx_bob.role);
  Send(server_psi_init);

  //  -------------------------- 4. client receive serverPsiInit -----------------------
  ServerPSIInit server_psi_init_recv;
  Recv(&server_psi_init_recv);
  psi_ctx_alice.SetRole(server_psi_init_recv.self_role(), server_psi_init_recv.self_size());
  if (psi_ctx_alice.peer_role != server_psi_init_recv.self_role() &&
      psi_ctx_alice.self_num != server_psi_init_recv.self_size()) {
    MS_LOG(WARNING) << "Context role set ERROR, please check!";
  }
  MS_LOG(INFO) << "SET PSI_CTX over";
  if (psi_ctx_alice.psi_type == "filter_ecdh") {
    ret = RunInverseFilterEcdhPsi(psi_ctx_alice, psi_ctx_bob);
  } else {
    MS_LOG(INFO) << "The psi protocol is not supported currently.";
  }
  return ret;
}

std::vector<std::string> RunInverseFilterEcdhPsi(const std::string &target_server_name, const PsiCtx &psi_ctx) {
  std::vector<std::string> align_results_vector;
  auto &verticalServer = VerticalServer::GetInstance();
  MS_LOG(INFO) << "Start hash input...";
  std::vector<std::string> input_hash_vct = HashInputs(psi_ctx.input_vct, psi_ctx.thread_num, psi_ctx.chunk_size);

  if (psi_ctx.role == "alice") {
    MS_LOG(INFO) << "[offline] Alice start computing p1^a...";
    auto p_a_vct = psi_ctx.ecc->HashToCurveAndMul(input_hash_vct, psi_ctx.compress_length, LENGTH_32);
    std::vector<std::string>().swap(input_hash_vct);
    BloomFilter bf_alice(p_a_vct, psi_ctx.thread_num, psi_ctx.neg_log_fp_rate);
    std::vector<std::string>().swap(p_a_vct);

    MS_LOG(INFO) << "----------------------- 2. alice receive bob_p_b -----------------------";
    BobPb bob_p_b_recv;
    verticalServer.Receive(target_server_name, &bob_p_b_recv);
    MS_LOG(INFO) << "Alice start decompress and compute p2^b^a --------------------------";
    auto p_b_a_vct = psi_ctx.ecc->DcpsAndMul(bob_p_b_recv.p_b_vct(), psi_ctx.compress_length, psi_ctx.compress_length);
    bob_p_b_recv.set_empty();

    MS_LOG(INFO) << " -------------------------- 3. alice send AlicePbaAndBFProto ------------------------";
    AlicePbaAndBF alice_p_b_a_bf(psi_ctx.bin_id, p_b_a_vct, bf_alice.GetData());
    verticalServer.Send(target_server_name, alice_p_b_a_bf);

    MS_LOG(INFO) << "-------------------------- 6. alice receive align_result -----------------------";
    std::vector<std::string> wrong_vct;
    std::vector<std::string> fix_vct;
    BobAlignResult bob_align_result_recv;
    verticalServer.Receive(target_server_name, &bob_align_result_recv);
    FindWrong(psi_ctx, bob_align_result_recv.align_result(), &wrong_vct, &fix_vct);
    bob_align_result_recv.set_empty();

    MS_LOG(INFO) << "-------------------------- 7. alice send wrong_id -----------------------";
    AliceCheck alice_check(psi_ctx.bin_id, wrong_vct.size(), wrong_vct);
    verticalServer.Send(target_server_name, alice_check);
    return fix_vct;
  } else {
    MS_LOG(INFO) << "[offline] Bob start computing p2^b...";
    auto p_b_vct = psi_ctx.ecc->HashToCurveAndMul(input_hash_vct, psi_ctx.compress_length, psi_ctx.compress_length);
    std::vector<std::string>().swap(input_hash_vct);
    MS_LOG(INFO) << "-------------------------- 1. bob send bobPb -----------------------";
    BobPb bob_p_b(psi_ctx.bin_id, p_b_vct);
    verticalServer.Send(target_server_name, bob_p_b);

    MS_LOG(INFO) << "-------------------------- 4. bob receive alice_p_b_a_bf -----------------------";
    AlicePbaAndBF alice_p_b_a_bf_recv;
    verticalServer.Receive(target_server_name, &alice_p_b_a_bf_recv);
    MS_LOG(INFO) << "Bob start decompress and compute p2^b^a^(b^-1) --------------------------";
    auto p_b_a_bI_vct =
      psi_ctx.ecc->DcpsAndInverseMul(alice_p_b_a_bf_recv.p_b_a_vct(), psi_ctx.compress_length, LENGTH_32);
    BloomFilter bf_alice_recv(alice_p_b_a_bf_recv.bf_alice(), psi_ctx.peer_num, psi_ctx.neg_log_fp_rate);
    alice_p_b_a_bf_recv.set_empty();
    align_results_vector = Align(p_b_a_bI_vct, bf_alice_recv, psi_ctx);
    std::vector<std::string>().swap(p_b_a_bI_vct);

    time_t time_start;
    time_t time_end;
    time(&time_start);
    std::sort(align_results_vector.begin(), align_results_vector.end());
    time(&time_end);
    MS_LOG(INFO) << "Bob sort align result, time cost: " << difftime(time_end, time_start) << " s.";

    MS_LOG(INFO) << "-------------------------- 5. bob send align_result -----------------------";
    BobAlignResult bob_align_result(psi_ctx.bin_id, align_results_vector);
    verticalServer.Send(target_server_name, bob_align_result);

    MS_LOG(INFO) << "-------------------------- 8. bob receive wrong_id -----------------------";
    AliceCheck alice_check_recv;
    verticalServer.Receive(target_server_name, &alice_check_recv);
    DelWrong(&align_results_vector, alice_check_recv.wrong_id());
    return align_results_vector;
  }
}

std::vector<std::string> RunPSI(const std::vector<std::string> &input_vct, const std::string &comm_role,
                                const std::string &target_server_name, size_t bin_id, size_t thread_num) {
  std::vector<std::string> ret;
  auto &verticalServer = VerticalServer::GetInstance();
  verticalServer.StartVerticalCommunicator();

  MS_LOG(INFO) << "Start RunPSICommunicateTest, init psi context...";
  PsiCtx psi_ctx;
  psi_ctx.bin_id = bin_id;
  psi_ctx.thread_num = thread_num;
  psi_ctx.input_vct = input_vct;
  psi_ctx.self_num = psi_ctx.input_vct.size();
  psi_ctx.ecc = std::make_unique<ECC>(psi_ctx.curve_name, psi_ctx.thread_num, psi_ctx.chunk_size);

  if (comm_role == "client") {
    MS_LOG(INFO) << "-------------------------- 1. client send clientPsiInit -----------------------";
    ClientPSIInit client_psi_init(psi_ctx.bin_id, psi_ctx.psi_type, psi_ctx.self_num);
    verticalServer.Send(target_server_name, client_psi_init);
    MS_LOG(INFO) << "-------------------------- 4. client receive serverPsiInit -----------------------";
    ServerPSIInit server_psi_init_recv;
    verticalServer.Receive(target_server_name, &server_psi_init_recv);
    if (server_psi_init_recv.bin_id() != psi_ctx.bin_id) {
      MS_LOG(ERROR) << "The bin_id is not same, please check bin_id: " << server_psi_init_recv.bin_id();
      return ret;
    }
    psi_ctx.SetRole(server_psi_init_recv.self_role(), server_psi_init_recv.self_size());
  } else if (comm_role == "server") {
    MS_LOG(INFO) << "-------------------------- 2. server receive clientPsiInit -----------------------";
    ClientPSIInit client_psi_init_recv;
    verticalServer.Receive(target_server_name, &client_psi_init_recv);
    psi_ctx.SetRole(client_psi_init_recv.self_size());
    MS_LOG(INFO) << "-------------------------- 3. server send serverPsiInit -----------------------";
    ServerPSIInit server_psi_init(psi_ctx.bin_id, psi_ctx.self_num, psi_ctx.role);
    verticalServer.Send(target_server_name, server_psi_init);
    if (client_psi_init_recv.bin_id() != psi_ctx.bin_id) {
      MS_LOG(ERROR) << "The bin_id is not same, please check bin_id: " << client_psi_init_recv.bin_id();
      return ret;
    }
  } else {
    MS_LOG(ERROR) << "Unknown communication role, wrong input role is " << comm_role;
    return ret;
  }

  if (!psi_ctx.CheckPsiCtxOK()) {
    MS_LOG(ERROR) << "Set PSI CTX ERROR!";
    return ret;
  }
  MS_LOG(INFO) << "Set PSI_CTX over, start computing...";

  if (psi_ctx.psi_type == "filter_ecdh") {
    ret = RunInverseFilterEcdhPsi(target_server_name, psi_ctx);
  } else {
    MS_LOG(INFO) << "The psi protocol is not supported currently.";
  }
  return ret;
}
}  // namespace psi
}  // namespace fl
}  // namespace mindspore
