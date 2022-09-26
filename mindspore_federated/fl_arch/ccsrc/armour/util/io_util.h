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

#ifndef MINDSPORE_FEDERATED_IO_UTIL_H
#define MINDSPORE_FEDERATED_IO_UTIL_H

#include <fstream>
#include <cstring>
#include <sstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "armour/base_crypto/bloom_filter.h"
#include "common/utils/log_adapter.h"
#include "common/protos/data_join.pb.h"

namespace mindspore {
namespace fl {
namespace psi {
std::vector<std::string> ReadFile(const std::string &csv_path);

void WriteFile(const std::string &csv_path, const std::vector<std::string> &in_data);

std::vector<std::string> ReadFileAndDeserialize(const std::string &csv_path, const size_t &input_num,
                                                const size_t &item_length);

void FlattenAndWriteFile(const std::string &out_path, const std::vector<std::string> &in_data);

BloomFilter ReadFileAndBuildFilter(const std::string &csv_path, const size_t &input_num, const int &neg_log_fp_rate);

std::string ReadBinFile(const std::string &csv_path);

void FilterWriteFile(const std::string &out_path, const std::string &ret);

std::vector<std::string> CreateFakeDataset(size_t begin, size_t size);

void GenDataSet();

struct ClientPSIInit {
 public:
  ~ClientPSIInit() = default;
  ClientPSIInit() = default;
  ClientPSIInit(const size_t &bin_id, const std::string &psi_type, const size_t &self_size)
      : bin_id_(bin_id), psi_type_(psi_type), self_size_(self_size) {}

  void set_bin_id(const size_t &bin_id) { bin_id_ = bin_id; }
  size_t bin_id() const { return bin_id_; }

  void set_psi_type(const std::string &psi_type_string) { psi_type_ = psi_type_string; }
  std::string psi_type() const { return psi_type_; }

  void set_self_size(const size_t &self_size_input) { self_size_ = self_size_input; }
  size_t self_size() const { return self_size_; }

 private:
  size_t bin_id_ = 0;
  std::string psi_type_ = "filter_ecdh";
  size_t self_size_ = 0;
};

bool Send(const ClientPSIInit &client_psi_init);

void Recv(ClientPSIInit *client_psi_init);

struct ServerPSIInit {
 public:
  ~ServerPSIInit() = default;
  ServerPSIInit() = default;
  ServerPSIInit(const size_t &bin_id, const size_t &self_size, const std::string &self_role)
      : bin_id_(bin_id), self_size_(self_size), self_role_(self_role) {}

  void set_bin_id(const size_t &bin_id) { bin_id_ = bin_id; }
  size_t bin_id() const { return bin_id_; }

  void set_self_size(const size_t &self_size_input) { self_size_ = self_size_input; }
  size_t self_size() const { return self_size_; }

  void set_self_role(const std::string &self_role_string) { self_role_ = self_role_string; }
  std::string self_role() const { return self_role_; }

 private:
  size_t bin_id_ = 0;
  size_t self_size_ = 0;
  std::string self_role_ = "alice";
};

bool Send(const ServerPSIInit &server_psi_init);

void Recv(ServerPSIInit *server_psi_init);

struct BobPb {
 public:
  ~BobPb() = default;
  BobPb() = default;
  BobPb(const size_t &bin_id, const std::vector<std::string> &p_b_vct) : bin_id_(bin_id) { p_b_vct_ = p_b_vct; }

  void set_bin_id(const size_t &bin_id) { bin_id_ = bin_id; }
  size_t bin_id() const { return bin_id_; }

  void set_p_b_vct(const std::vector<std::string> &p_b_vct) { p_b_vct_ = p_b_vct; }
  std::vector<std::string> p_b_vct() const { return p_b_vct_; }

 private:
  size_t bin_id_ = 0;
  std::vector<std::string> p_b_vct_;
};

bool Send(const BobPb &bob_p_b);

void Recv(BobPb *bob_p_b);

struct AlicePbaAndBF {
 public:
  ~AlicePbaAndBF() = default;
  AlicePbaAndBF() = default;
  AlicePbaAndBF(const size_t &bin_id, const std::vector<std::string> &p_b_a_vct, const std::string &bf_alice)
      : bin_id_(bin_id), bf_alice_(bf_alice) {
    p_b_a_vct_ = p_b_a_vct;
  }

  void set_bin_id(const size_t &bin_id) { bin_id_ = bin_id; }
  size_t bin_id() const { return bin_id_; }

  void set_p_b_a_vct(const std::vector<std::string> &p_b_a_vct) { p_b_a_vct_ = p_b_a_vct; }
  std::vector<std::string> p_b_a_vct() const { return p_b_a_vct_; }

  void set_bf_alice(const std::string &bf_alice) { bf_alice_ = bf_alice; }
  std::string bf_alice() const { return bf_alice_; }

 private:
  size_t bin_id_ = 0;
  std::vector<std::string> p_b_a_vct_;
  std::string bf_alice_;
};

bool Send(const AlicePbaAndBF &alice_pba_bf);

void Recv(AlicePbaAndBF *alice_p_b_a_bf);

struct BobAlignResult {
 public:
  ~BobAlignResult() = default;
  BobAlignResult() = default;
  BobAlignResult(const size_t &bin_id, const std::vector<std::string> &align_result) : bin_id_(bin_id) {
    align_result_ = align_result;
  }

  void set_bin_id(const size_t &bin_id) { bin_id_ = bin_id; }
  size_t bin_id() const { return bin_id_; }

  void set_align_resul(const std::vector<std::string> &align_result) { align_result_ = align_result; }
  std::vector<std::string> align_result() const { return align_result_; }

 private:
  size_t bin_id_ = 0;
  std::vector<std::string> align_result_;
};

bool Send(const BobAlignResult &bob_align_result);

void Recv(BobAlignResult *bob_align_result);

struct AliceCheck {
 public:
  ~AliceCheck() = default;
  AliceCheck() = default;
  AliceCheck(const size_t &bin_id, const size_t &wrong_num, const std::vector<std::string> &wrong_id)
      : bin_id_(bin_id), wrong_num_(wrong_num) {
    wrong_id_ = wrong_id;
  }

  void set_bin_id(const size_t &bin_id) { bin_id_ = bin_id; }
  size_t bin_id() const { return bin_id_; }

  void set_wrong_num(const size_t &wrong_num) { wrong_num_ = wrong_num; }
  size_t wrong_num() const { return wrong_num_; }

  void set_wrong_id(const std::vector<std::string> &wrong_id) { wrong_id_ = wrong_id; }
  std::vector<std::string> wrong_id() const { return wrong_id_; }

 private:
  size_t bin_id_ = 0;
  size_t wrong_num_ = 0;
  std::vector<std::string> wrong_id_;
};

bool Send(const AliceCheck &alice_check);

void Recv(AliceCheck *alice_check);
}  // namespace psi
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_FEDERATED_IO_UTIL_H
