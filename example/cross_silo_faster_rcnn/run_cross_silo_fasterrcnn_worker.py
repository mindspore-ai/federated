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
"""start running lenet worker of cross silo mode"""

import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run run_cross_silo_fasterrcnn_worker..py case")
parser.add_argument("--yaml_config", type=str, default="default_yaml_config.yaml")

parser.add_argument("--device_target", type=str, default="GPU")
parser.add_argument("--fl_iteration_num", type=int, default=30)
parser.add_argument("--client_batch_size", type=int, default=32)
parser.add_argument("--client_learning_rate", type=float, default=0.01)
parser.add_argument("--local_worker_num", type=int, default=4)
parser.add_argument("--dataset_path", type=str, default="")
parser.add_argument("--sync_type", type=str, default="fixed", choices=["fixed", "adaptive"])
parser.add_argument("--http_server_address", type=str, default="127.0.0.1:5555")
parser.add_argument("--client_epoch_num", type=int, default=1)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--pre_trained", type=str, default="")

args, _ = parser.parse_known_args()
yaml_config = args.yaml_config
device_target = args.device_target
fl_iteration_num = args.fl_iteration_num
client_batch_size = args.client_batch_size
client_learning_rate = args.client_learning_rate
local_worker_num = args.local_worker_num
dataset_path = args.dataset_path
sync_type = args.sync_type
http_server_address = args.http_server_address
client_epoch_num = args.client_epoch_num
pre_trained = args.pre_trained

cur_dir = os.path.dirname(os.path.abspath(__file__))
yaml_config = os.path.join(cur_dir, yaml_config)

assert local_worker_num > 0, "The local worker number should not <= 0."

for i in range(local_worker_num):
    device_id = args.device_id + i
    cmd_worker = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && "
    cmd_worker += "rm -rf ${execute_path}/logs/worker_" + str(i) + "/ &&"
    cmd_worker += "mkdir -p ${execute_path}/logs/worker_" + str(i) + "/ &&"
    cmd_worker += "cd ${execute_path}/logs/worker_" + str(i) + "/ || exit && export GLOG_v=1 && "
    cmd_worker += "python ${self_path}/../../test_fl_fasterrcnn.py"
    cmd_worker += " --ms_role=MS_WORKER"
    cmd_worker += " --yaml_config=" + str(yaml_config)
    cmd_worker += " --device_target=" + device_target
    cmd_worker += " --fl_iteration_num=" + str(fl_iteration_num)
    cmd_worker += " --dataset_path=" + str(dataset_path)
    cmd_worker += " --user_id=" + str(i)
    cmd_worker += " --sync_type=" + sync_type
    cmd_worker += " --http_server_address=" + http_server_address
    cmd_worker += " --pre_trained=" + pre_trained
    cmd_worker += " --client_epoch_num=" + str(client_epoch_num)
    cmd_worker += " --device_id=" + str(device_id)
    cmd_worker += " > worker.log 2>&1 &"

    subprocess.call(['bash', '-c', cmd_worker])
