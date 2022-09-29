# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""psi with communication st"""

import argparse
from mindspore_federated._mindspore_federated import RunPSI


def get_parser():
    """parser argument"""
    parser = argparse.ArgumentParser(description="Run PSI with Communication")

    parser.add_argument("--comm_role", type=str, default="server")
    parser.add_argument("--input_begin", type=int, default=1)
    parser.add_argument("--input_end", type=int, default=1000)
    parser.add_argument("--read_file", type=bool, default=False)
    parser.add_argument("--file_name", type=str, default="null")
    parser.add_argument("--domain_name", type=str, default="127.0.0.1:8004")
    parser.add_argument("--peer_domain_name", type=str, default="127.0.0.1:8005")
    parser.add_argument("--bin_id", type=int, default=1)
    parser.add_argument("--thread_num", type=int, default=0)
    parser.add_argument("--need_check", type=bool, default=False)
    parser.add_argument("--peer_input_begin", type=int, default=1)
    parser.add_argument("--peer_input_end", type=int, default=1000)
    parser.add_argument("--peer_read_file", type=bool, default=False)
    parser.add_argument("--peer_file_name", type=str, default="null")
    return parser


args, _ = get_parser().parse_known_args()

comm_role = args.comm_role
input_begin = args.input_begin
input_end = args.input_end
read_file = args.read_file
file_name = args.file_name
domain_name = args.domain_name
peer_domain_name = args.peer_domain_name
bin_id = args.bin_id
thread_num = args.thread_num
need_check = args.need_check
peer_input_begin = args.peer_input_begin
peer_input_end = args.peer_input_end
peer_read_file = args.peer_read_file
peer_file_name = args.peer_file_name


def compute_right_result(self_input, peer_input):
    self_input_set = set(self_input)
    peer_input_set = set(peer_input)
    return self_input_set.intersection(peer_input_set)


def check_psi(actual_result_, psi_result_):
    actual_result_set = set(actual_result_)
    psi_result_set = set(psi_result_)
    wrong_num = len(psi_result_set.difference(actual_result_set).union(actual_result_set.difference(psi_result_set)))
    return wrong_num


def generate_input_data(input_begin_, input_end_, read_file_, file_name_):
    input_data_ = []
    if read_file_:
        with open(file_name_, 'r') as f:
            for line in f.readlines():
                input_data_.append(line.strip())
    else:
        input_data_ = [str(i) for i in range(input_begin_, input_end_)]
    return input_data_


if __name__ == "__main__":
    input_data = generate_input_data(input_begin, input_end, read_file, file_name)
    bin_size = 2
    for bin_id in range(1, 1 + bin_size):
        psi_result = RunPSI(input_data, comm_role, domain_name, peer_domain_name, thread_num, bin_id)
        print("PSI result:{}".format(psi_result[0:20]))
        if need_check:
            peer_input_data = generate_input_data(peer_input_begin, peer_input_end, peer_read_file, peer_file_name)
            actual_result = compute_right_result(input_data, peer_input_data)
            if check_psi(actual_result, psi_result) == 0:
                print("success, PSI check pass!")
            else:
                print("ERROR, PSI check failed!")
