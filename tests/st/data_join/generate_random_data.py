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
import pandas as pd
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description="Run generate_random_data.py case")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="vfl/datasets/test_data_0.csv")
    parser.add_argument("--data_num_per_file", type=int, default=10000)
    parser.add_argument("--id_len", type=int, default=20)
    parser.add_argument("--feature_num", type=int, default=30)
    return parser


if __name__ == '__main__':
    args, _ = get_parser().parse_known_args()
    for key in args.__dict__:
        print('[', key, ']', args.__dict__[key], flush=True)
    seed = args.seed
    output_path = args.output_path
    data_num_per_file = args.data_num_per_file
    id_len = args.id_len
    feature_num = args.feature_num

    np.random.seed(seed)
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'

    output = ["oaid" + ''.join([",feature{}".format(_) for _ in range(feature_num)])]
    oaid_list = list()

    for i in range(data_num_per_file):
        random_str = ""
        length = len(base_str) - 1
        for j in range(id_len):
            random_str += base_str[np.random.randint(0, length)]
        if random_str in oaid_list:
            print("oaid: {} is existed".format(random_str))
            continue
        oaid_list.append(random_str)
        for j in range(feature_num):
            random_str += ",{}".format(np.random.randn())
        output.append(random_str)
    print('-' * 1000)
    output = '\n'.join(output)
    print(output)
    print('+' * 1000)
    with open(output_path, "w") as f:
        f.write(output)

    test_data = pd.read_csv(output_path)
    print(test_data)
