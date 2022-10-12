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
from mindspore_federated.data_join.worker import FLDataWorker


def get_parser():
    """
    Get parser.
    """
    parser = argparse.ArgumentParser(description="Run run_data_join.py case")
    parser.add_argument("--role", type=str, default="leader")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8004")
    parser.add_argument("--peer_server_address", type=str, default="127.0.0.1:8005")
    parser.add_argument("--worker_config_path", type=str, default="vfl/leader.yaml")
    parser.add_argument("--schema_path", type=str, default="vfl/schema.yaml")
    return parser


if __name__ == '__main__':
    args, _ = get_parser().parse_known_args()
    for key in args.__dict__:
        print('[', key, ']', args.__dict__[key], flush=True)
    role = args.role
    worker_config_path = args.worker_config_path
    schema_path = args.schema_path
    server_address = args.server_address
    peer_server_address = args.peer_server_address
    worker = FLDataWorker(role=role,
                          worker_config_path=worker_config_path,
                          data_schema_path=schema_path,
                          server_address=server_address,
                          peer_server_address=peer_server_address
                          )
    worker.export()
