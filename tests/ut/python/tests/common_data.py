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

import yaml
import numpy as np


def generate_random_data(seed=0,
                         leader_output_path="temp/leader_data_*.csv",
                         follower_output_path="temp/follower_data_*.csv",
                         intersection_output_path="temp/intersection_data.csv",
                         leader_file_num=4,
                         follower_file_num=2,
                         leader_data_num=300,
                         follower_data_num=200,
                         overlap_num=100,
                         id_len=20,
                         feature_num=30,
                         ):
    """generate random data"""
    np.random.seed(seed)
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'

    output = list()
    oaid_list = list()
    type_list = ["int", "float", "string"] * (feature_num // 3 + 1)
    for i in range(leader_data_num + follower_data_num - overlap_num):
        while True:
            random_str = ""
            length = len(base_str) - 1
            for _ in range(id_len):
                random_str += base_str[np.random.randint(0, length)]
            if random_str not in oaid_list:
                break
        oaid_list.append(random_str)
        for j in range(feature_num):
            if type_list[j] == "int":
                random_str += ",{}".format(np.random.randint(-100, 100))
            elif type_list[j] == "float":
                random_str += ",{}".format(np.random.randn())
            elif type_list[j] == "string":
                random_str += ",{}".format(oaid_list[-1])
            else:
                random_str += ","
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


def generate_schema(
        yaml_path="temp/schema.yaml",
        **kwargs,
):
    """generate schema"""
    schema = dict()
    for key in kwargs:
        schema[key] = {"type": kwargs[key]}
    with open(yaml_path, "w") as f:
        yaml.dump(data=schema, stream=f)
