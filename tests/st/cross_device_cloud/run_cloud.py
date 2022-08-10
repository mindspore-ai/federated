# Copyright 2021 Huawei Technologies Co., Ltd
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
"""start running lenet cross device cloud mode"""

import os
import sys
import argparse
import numpy as np
from mindspore_federated.startup.ssl_config import SSLConfig

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

parser = argparse.ArgumentParser(description="test_fl_cloud")

parser.add_argument("--ms_role", type=str, default="MS_WORKER")
# common
parser.add_argument("--yaml_config", type=str, default="default_yaml_config.yaml")
# for server
parser.add_argument("--http_server_address", type=str, default=2)
parser.add_argument("--tcp_server_ip", type=str, default="127.0.0.1")
parser.add_argument("--checkpoint_dir", type=str, default="./fl_ckpt/")

# for scheduler
parser.add_argument("--scheduler_manage_address", type=str, default="127.0.0.1:11202")

args, _ = parser.parse_known_args()

ssl_config = SSLConfig(server_password="server_password_12345", client_password="client_password_12345")


def get_trainable_params(network):
    """get trainable params"""
    feature_map = {}
    for param in network.trainable_params():
        param_np = param.asnumpy()
        if param_np.dtype != np.float32:
            continue
        feature_map[param.name] = param_np
    return feature_map


def start_one_server():
    """start one server"""
    from network.lenet import LeNet5
    from mindspore_federated import FLServerJob

    yaml_config = args.yaml_config
    tcp_server_ip = args.tcp_server_ip
    http_server_address = args.http_server_address
    checkpoint_dir = args.checkpoint_dir

    network = LeNet5(62, 3)
    job = FLServerJob(yaml_config=yaml_config, http_server_address=http_server_address, tcp_server_ip=tcp_server_ip,
                      checkpoint_dir=checkpoint_dir, ssl_config=ssl_config)
    feature_map = get_trainable_params(network)
    job.run(feature_map)


def start_one_scheduler():
    """start one scheduler"""
    from mindspore_federated import FlSchedulerJob

    yaml_config = args.yaml_config
    scheduler_manage_address = args.scheduler_manage_address

    job = FlSchedulerJob(yaml_config=yaml_config, manage_address=scheduler_manage_address, ssl_config=ssl_config)
    job.run()


if __name__ == "__main__":
    ms_role = args.ms_role
    if ms_role == "MS_SERVER":
        start_one_server()
    elif ms_role == "MS_SCHED":
        start_one_scheduler()
