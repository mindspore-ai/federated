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
import ast
import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run test_cross_silo_femnist.py case")
parser.add_argument("--yaml_config", type=str, default="default_yaml_config.yaml")

parser.add_argument("--device_target", type=str, default="GPU")
parser.add_argument("--fl_iteration_num", type=int, default=25)
parser.add_argument("--client_batch_size", type=int, default=32)
parser.add_argument("--client_learning_rate", type=float, default=0.01)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--dataset_path", type=str, default="")
parser.add_argument("--sync_type", type=str, default="fixed", choices=["fixed", "adaptive"])
parser.add_argument("--http_server_address", type=str, default="127.0.0.1:5555")
parser.add_argument("--device_num", type=int, default=2)
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False)

args, _ = parser.parse_known_args()
yaml_config = args.yaml_config
device_target = args.device_target
fl_iteration_num = args.fl_iteration_num
client_batch_size = args.client_batch_size
client_learning_rate = args.client_learning_rate
dataset_path = args.dataset_path
sync_type = args.sync_type
http_server_address = args.http_server_address
device_id = args.device_id
device_num = args.device_num
run_distribute = args.run_distribute

cur_dir = os.path.dirname(os.path.abspath(__file__))
yaml_config = os.path.join(cur_dir, yaml_config)

cmd_worker = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && "
cmd_worker += "rm -rf ${execute_path}/logs/worker_distributed" + "/ &&"
cmd_worker += "mkdir -p ${execute_path}/logs/worker_distributed" + "/ &&"
cmd_worker += "cd ${execute_path}/logs/worker_distributed" + "/ || exit && export GLOG_v=1 && "
cmd_worker += "mpirun --allow-run-as-root -n " + str(device_num) + " --output-filename log_output "
cmd_worker += " --merge-stderr-to-stdout python ${self_path}/../../test_cross_silo_femnist.py"
cmd_worker += " --ms_role=MS_WORKER"
cmd_worker += " --yaml_config=" + str(yaml_config)
cmd_worker += " --device_target=" + device_target
cmd_worker += " --fl_iteration_num=" + str(fl_iteration_num)
cmd_worker += " --dataset_path=" + str(dataset_path)
cmd_worker += " --sync_type=" + sync_type
cmd_worker += " --client_batch_size=" + str(client_batch_size)
cmd_worker += " --client_learning_rate=" + str(client_learning_rate)
cmd_worker += " --http_server_address=" + http_server_address
cmd_worker += " --run_distribute=" + str(run_distribute)
cmd_worker += " --device_num=" + str(device_num)
cmd_worker += " --device_id=" + str(device_id)
cmd_worker += " > worker.log 2>&1 &"

subprocess.call(['bash', '-c', cmd_worker])
