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

psi::BobPb ParseBobPbProto(datajoin::BobPbProto bobPbProto);

psi::ClientPSIInit ParseClientPSIInitProto(datajoin::ClientPSIInitProto clientPSIInitProto);

psi::ServerPSIInit ParseServerPSIInitProto(datajoin::ServerPSIInitProto serverPSIInitProto);

psi::AlicePbaAndBF ParseAlicePbaAndBFProto(datajoin::AlicePbaAndBFProto alicePbaAndBFProto);

psi::BobAlignResult ParseBobAlignResultProto(datajoin::BobAlignResultProto bobAlignResultProto);

psi::AliceCheck ParseAliceCheckProto(datajoin::AliceCheckProto aliceCheckProto);

psi::PlainData ParsePlainDataProto(datajoin::PlainDataProto plainDataProto);
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_ARCH_CCSRC_VERTICAL_UTILS_PSI_UTILS_H_
