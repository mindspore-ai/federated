# Copyright 2023 Huawei Technologies Co., Ltd
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
"""start run cross silo worker"""
import os
import argparse
import subprocess


cur_dir = os.getcwd()
log_dir = './logs/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
code_name = 'test_cross_silo_luna16'

parser = argparse.ArgumentParser(description="Run worker case")
parser.add_argument("--yaml_config", type=str, default="default_yaml_config.yaml")
parser.add_argument("--device_target", type=str, default="GPU")
parser.add_argument("--fl_iteration_num", type=int, default=5000)
parser.add_argument("--client_batch_size", type=int, default=32)
parser.add_argument("--client_learning_rate", type=float, default=0.001)
parser.add_argument("--worker_num", type=int, default=3)
parser.add_argument("--data_dir", type=str, default="./luna16/train/")
parser.add_argument("--sync_type", type=str, default="fixed", choices=["fixed", "adaptive"])
parser.add_argument("--http_server_address", type=str, default="127.0.0.1:3231")
parser.add_argument("--local_epoch", type=int, default=20)

args, _ = parser.parse_known_args()
yaml_config = args.yaml_config
device_target = args.device_target
fl_iteration_num = args.fl_iteration_num
client_batch_size = args.client_batch_size
client_learning_rate = args.client_learning_rate
worker_num = args.worker_num
data_dir = args.data_dir
sync_type = args.sync_type
http_server_address = args.http_server_address
local_epoch = args.local_epoch

yaml_config = os.path.join(cur_dir, yaml_config)

for i in range(worker_num):
    cmd_worker = "rm -rf {}/logs/worker_{}/ && ".format(cur_dir, i)
    cmd_worker += " mkdir {}/logs/worker_{}/ && ".format(cur_dir, i)
    cmd_worker += " export GLOG_v=2 && export DEVICE_ID={} && ".format(i%2 + 2) # i%2
    cmd_worker += " python {}/{}.py".format(cur_dir, code_name)
    cmd_worker += " --ms_role=MS_WORKER"
    cmd_worker += " --yaml_config={}".format(yaml_config)
    cmd_worker += " --device_target={}".format(device_target)
    cmd_worker += " --fl_iteration_num={}".format(fl_iteration_num)
    cmd_worker += " --client_batch_size={}".format(client_batch_size)
    cmd_worker += " --client_learning_rate={}".format(client_learning_rate)
    cmd_worker += " --data_dir={}".format(os.path.join(data_dir, str(i)))
    cmd_worker += " --sync_type={}".format(sync_type)
    cmd_worker += " --device_id={}".format(i%2 + 2)
    cmd_worker += " --http_server_address={}".format(http_server_address)
    cmd_worker += " --local_epoch={}".format(local_epoch)
    cmd_worker += " --ckpt_dir={}/logs/worker_{}/".format(cur_dir, i)
    cmd_worker += " > {}/logs/worker_{}/worker.log 2>&1 &".format(cur_dir, i)

    subprocess.call(['bash', '-c', cmd_worker])
