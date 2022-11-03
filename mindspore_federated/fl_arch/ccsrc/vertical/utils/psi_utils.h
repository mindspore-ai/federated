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
#ifndef MINDSPORE_FL_ARCH_CCSRC_VERTICAL_UTILS_PSI_UTILS_H_
#define MINDSPORE_FL_ARCH_CCSRC_VERTICAL_UTILS_PSI_UTILS_H_

#include <vector>
#include <string>
#include "armour/util/io_util.h"
#include "vertical/common.h"

namespace mindspore {
namespace fl {
void CreateBobPbProto(datajoin::BobPbProto *bob_p_b_proto, const psi::BobPb &bob_p_b);

void CreateClientPSIInitProto(datajoin::ClientPSIInitProto *client_init_proto, const psi::ClientPSIInit &client_init);

void CreateAlicePbaAndBFProto(datajoin::AlicePbaAndBFProto *alice_pba_bf_proto, const psi::AlicePbaAndBF &alice_pba_bf);

void CreateServerPSIInitProto(datajoin::ServerPSIInitProto *server_init_proto, const psi::ServerPSIInit &server_init);

void CreateBobAlignResultProto(datajoin::BobAlignResultProto *bob_align_result_proto,
                               const psi::BobAlignResult &bob_align_result);

void CreateAliceCheckProto(datajoin::AliceCheckProto *alice_check_proto, const psi::AliceCheck &alice_check);

void CreatePlainDataProto(datajoin::PlainDataProto *plain_data_proto, const psi::PlainData &plain_data);

psi::BobPb ParseBobPbProto(const datajoin::BobPbProto &bobPbProto);

psi::ClientPSIInit ParseClientPSIInitProto(const datajoin::ClientPSIInitProto &clientPSIInitProto);

psi::ServerPSIInit ParseServerPSIInitProto(const datajoin::ServerPSIInitProto &serverPSIInitProto);

psi::AlicePbaAndBF ParseAlicePbaAndBFProto(const datajoin::AlicePbaAndBFProto &alicePbaAndBFProto);

psi::BobAlignResult ParseBobAlignResultProto(const datajoin::BobAlignResultProto &bobAlignResultProto);

psi::AliceCheck ParseAliceCheckProto(const datajoin::AliceCheckProto &aliceCheckProto);

psi::PlainData ParsePlainDataProto(const datajoin::PlainDataProto &plainDataProto);

std::vector<std::string> StringSplit(const std::string &str, char sign);

std::vector<std::string> GetSplitData(std::string *begin_ptr, size_t index, size_t data_size, const size_t slice_size);

size_t CalSliceCount(size_t data_size, const size_t slice_size);

void UpdateSliceData(std::string *proto_data, std::vector<uint8_t> *slice_data, size_t *index, std::string *offset,
                     const char &sign);

SliceProto CreateProtoWithSlices(const psi::AlicePbaAndBF &alice_pba_bf);

SliceProto CreateProtoWithSlices(const psi::AliceCheck &alice_check);

SliceProto CreateProtoWithSlices(const psi::BobPb &bob_pb);

SliceProto CreateProtoWithSlices(const psi::BobAlignResult &bob_align_result);

SliceProto CreateProtoWithSlices(const psi::PlainData &plain_data);

psi::AlicePbaAndBF ParseProtoWithSlices(const std::vector<datajoin::AlicePbaAndBFProto> &protos);

psi::AliceCheck ParseProtoWithSlices(const std::vector<datajoin::AliceCheckProto> &protos);

psi::BobPb ParseProtoWithSlices(const std::vector<datajoin::BobPbProto> &protos);

psi::BobAlignResult ParseProtoWithSlices(const std::vector<datajoin::BobAlignResultProto> &protos);

psi::PlainData ParseProtoWithSlices(const std::vector<datajoin::PlainDataProto> &protos);
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_UTILS_PSI_UTILS_H_
