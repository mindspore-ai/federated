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
"""Load joined data."""
import os
import argparse
from mindspore import dataset as ds
from mindspore_federated.data_join.io import load_mindrecord


def get_parser():
    parser = argparse.ArgumentParser(description="Run load_joined_data.py case")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="vfl/output/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--drop_remainder", type=bool, default=False)
    parser.add_argument("--shuffle", type=bool, default=True)
    return parser


if __name__ == "__main__":
    args, _ = get_parser().parse_known_args()
    for key in args.__dict__:
        print('[', key, ']', args.__dict__[key], flush=True)
    seed = args.seed
    data_dir = args.data_dir
    batch_size = args.batch_size
    drop_remainder = args.drop_remainder
    shuffle = args.shuffle

    ds.config.set_seed(seed)
    dataset_files = [data_dir + _ for _ in os.listdir(data_dir) if "db" not in _]
    dataset = load_mindrecord(dataset_files=dataset_files, batch_size=batch_size, drop_remainder=drop_remainder,
                              shuffle=shuffle)
    print("dataset size: ", dataset.get_dataset_size())
    print("batch size: ", dataset.get_batch_size())
    for key in dataset.create_dict_iterator():
        print(key)
