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
"""Run data join."""
import argparse
import os
from mindspore_federated.data_join import FLDataWorker


def mkdir(directory):
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass


def get_parser():
    """
    Get parser.
    """
    parser = argparse.ArgumentParser(description="Run run_data_join.py case")
    parser.add_argument("--role", type=str, default="leader")
    parser.add_argument("--main_table_files", type=str, default="vfl/input/leader/")
    parser.add_argument("--output_dir", type=str, default="vfl/output/leader/")
    parser.add_argument("--data_schema_path", type=str, default="vfl/leader_schema.yaml")
    parser.add_argument("--http_server_address", type=str, default="127.0.0.1:1086")
    parser.add_argument("--remote_server_address", type=str, default="127.0.0.1:1087")
    parser.add_argument("--primary_key", type=str, default="oaid")
    parser.add_argument("--bucket_num", type=int, default=5)
    parser.add_argument("--store_type", type=str, default="csv")
    parser.add_argument("--shard_num", type=int, default=1)
    parser.add_argument("--join_type", type=str, default="psi")
    parser.add_argument("--thread_num", type=int, default=0)
    return parser


if __name__ == '__main__':
    mkdir("vfl")
    mkdir("vfl/output")
    mkdir("vfl/output/leader")
    mkdir("vfl/output/follower")
    args, _ = get_parser().parse_known_args()
    for key in args.__dict__:
        print('[', key, ']', args.__dict__[key], flush=True)
    role = args.role
    main_table_files = args.main_table_files
    output_dir = args.output_dir
    data_schema_path = args.data_schema_path
    http_server_address = args.http_server_address
    remote_server_address = args.remote_server_address
    primary_key = args.primary_key
    bucket_num = args.bucket_num
    store_type = args.store_type
    shard_num = args.shard_num
    join_type = args.join_type
    thread_num = args.thread_num
    worker = FLDataWorker(role=role,
                          main_table_files=main_table_files,
                          output_dir=output_dir,
                          data_schema_path=data_schema_path,
                          http_server_address=http_server_address,
                          remote_server_address=remote_server_address,
                          primary_key=primary_key,
                          bucket_num=bucket_num,
                          store_type=store_type,
                          shard_num=shard_num,
                          join_type=join_type,
                          thread_num=thread_num,
                          )
    worker.export()
