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
"""Generate random data."""
import argparse
import os
import numpy as np


def mkdir(directory):
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass


def get_parser():
    """Get args."""
    parser = argparse.ArgumentParser(description="Run generate_random_data.py case")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_output_path", type=str, default="vfl/input/total_data.csv")
    parser.add_argument("--intersection_output_path", type=str, default="vfl/input/intersection_data.csv")
    parser.add_argument("--leader_output_path", type=str, default="vfl/input/leader/data_*.csv")
    parser.add_argument("--follower_output_path", type=str, default="vfl/input/follower/data_*.csv")
    parser.add_argument("--leader_file_num", type=int, default=4)
    parser.add_argument("--follower_file_num", type=int, default=2)
    parser.add_argument("--leader_data_num", type=int, default=300)
    parser.add_argument("--follower_data_num", type=int, default=200)
    parser.add_argument("--overlap_num", type=int, default=100)
    parser.add_argument("--id_len", type=int, default=20)
    parser.add_argument("--feature_num", type=int, default=30)
    return parser


if __name__ == '__main__':
    mkdir("vfl")
    mkdir("vfl/input")
    mkdir("vfl/input/leader")
    mkdir("vfl/input/follower")
    args, _ = get_parser().parse_known_args()
    for key in args.__dict__:
        print('[', key, ']', args.__dict__[key], flush=True)
    seed = args.seed
    total_output_path = args.total_output_path
    intersection_output_path = args.intersection_output_path
    leader_output_path = args.leader_output_path
    follower_output_path = args.follower_output_path
    leader_file_num = args.leader_file_num
    follower_file_num = args.follower_file_num
    leader_data_num = args.leader_data_num
    follower_data_num = args.follower_data_num
    overlap_num = args.overlap_num
    id_len = args.id_len
    feature_num = args.feature_num

    np.random.seed(seed)
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'

    output = list()
    oaid_list = list()

    for i in range(leader_data_num + follower_data_num - overlap_num):
        while True:
            random_str = ""
            length = len(base_str) - 1
            for j in range(id_len):
                random_str += base_str[np.random.randint(0, length)]
            if random_str not in oaid_list:
                break
        oaid_list.append(random_str)
        for j in range(feature_num):
            random_str += ",{}".format(np.random.randn())
        output.append(random_str)

    leader_output_list = [list() for _ in range(leader_file_num)]
    follower_output_list = [list() for _ in range(follower_file_num)]
    for i in range(leader_data_num):
        index = np.random.randint(0, leader_file_num)
        leader_output_list[index].append(output[i])
    for i in range(follower_data_num):
        index = np.random.randint(0, follower_file_num)
        follower_output_list[index].append(output[-1-i])

    column_name = "oaid" + ''.join([",feature{}".format(_) for _ in range(feature_num)]) + "\n"
    total_output = column_name + '\n'.join(output)
    with open(total_output_path, "w") as f:
        f.write(total_output)
    for i, leader_output in enumerate(leader_output_list):
        leader_output = column_name + '\n'.join(leader_output)
        with open(leader_output_path.replace("*", str(i)), "w") as f:
            f.write(leader_output)
    for i, follower_output in enumerate(follower_output_list):
        follower_output = column_name + '\n'.join(follower_output)
        with open(follower_output_path.replace("*", str(i)), "w") as f:
            f.write(follower_output)
    intersection_output_list = [list() for _ in range(1)]
    start_num = (leader_data_num + follower_data_num - overlap_num) - follower_data_num
    for i in range(overlap_num):
        index = np.random.randint(0, 1)
        intersection_output_list[index].append(output[start_num+i])

    for i, intersection_output in enumerate(intersection_output_list):
        intersection_output = column_name + '\n'.join(intersection_output)
        with open(intersection_output_path.replace("*", str(i)), "w") as f:
            f.write(intersection_output)
