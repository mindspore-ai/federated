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
from mindspore_federated import VerticalFederatedCommunicator, ServerConfig


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
    parser.add_argument("--role", type=str, default="leader",
                        help="Role of the worker, which must be set in both leader and follower. Supports"
                             "[leader, follower].")
    parser.add_argument("--main_table_files", type=str, default="vfl/input/leader/",
                        help="The raw data paths, which must be set in both leader and follower.")
    parser.add_argument("--output_dir", type=str, default="vfl/output/leader/",
                        help="The output directory, which must be set in both leader and follower.")
    parser.add_argument("--data_schema_path", type=str, default="vfl/leader_schema.yaml",
                        help="Path of data schema file, which must be set in both leader and follower. User need to"
                             "provide the column name and type of the data to be exported in the schema. The schema"
                             "needs to be parsed as a two-level key-value dictionary. The key of the first-level"
                             "dictionary is the column name, and the value is the second-level dictionary. The key of"
                             "the second-level dictionary must be a string: type, and the value is the type of the"
                             "exported data. Currently, the types support [int32, int64, float32, float64, string,"
                             "bytes].")
    parser.add_argument("--server_name", type=str, default="leader_node",
                        help="Local http server name.")
    parser.add_argument("--remote_server_name", type=str, default="follower_node",
                        help="Remote http server name.")
    parser.add_argument("--http_server_address", type=str, default="127.0.0.1:1086",
                        help="Local IP and Port Address, which must be set in both leader and follower.")
    parser.add_argument("--remote_server_address", type=str, default="127.0.0.1:1087",
                        help="Peer IP and Port Address, which must be set in both leader and follower.")
    parser.add_argument("--primary_key", type=str, default="oaid",
                        help="The primary key. The value set by leader is used, and the value set by follower is"
                             "invalid.")
    parser.add_argument("--bucket_num", type=int, default=5,
                        help="The number of buckets. The value set by leader is used, and the value set by follower is"
                             "invalid.")
    parser.add_argument("--store_type", type=str, default="csv", help="The data store type.")
    parser.add_argument("--shard_num", type=int, default=1,
                        help="The output number of each bucket when export. The value set by leader is used, and the"
                             "value set by follower is invalid.")
    parser.add_argument("--join_type", type=str, default="psi",
                        help="The data join type. The value set by leader is used, and the value set by follower is"
                             "invalid.")
    parser.add_argument("--thread_num", type=int, default=0, help="The thread number of psi.")
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
    server_name = args.server_name
    remote_server_name = args.remote_server_name
    http_server_address = args.http_server_address
    remote_server_address = args.remote_server_address
    primary_key = args.primary_key
    bucket_num = args.bucket_num
    store_type = args.store_type
    shard_num = args.shard_num
    join_type = args.join_type
    thread_num = args.thread_num
    http_server_config = ServerConfig(server_name=server_name, server_address=http_server_address)
    remote_server_config = ServerConfig(server_name=remote_server_name, server_address=remote_server_address)
    vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
                                                          remote_server_config=remote_server_config)
    vertical_communicator.launch()

    worker = FLDataWorker(role=role,
                          main_table_files=main_table_files,
                          output_dir=output_dir,
                          data_schema_path=data_schema_path,
                          primary_key=primary_key,
                          bucket_num=bucket_num,
                          store_type=store_type,
                          shard_num=shard_num,
                          join_type=join_type,
                          thread_num=thread_num,
                          communicator=vertical_communicator
                          )
    worker.export()
