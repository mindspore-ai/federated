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
"""start running lenet server of cross device cloud mode"""

import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run run_cloud.py case")

parser.add_argument("--yaml_config", type=str, default="default_yaml_config.yaml")
parser.add_argument("--tcp_server_ip", type=str, default="127.0.0.1")
parser.add_argument("--checkpoint_dir", type=str, default="./fl_ckpt/")
parser.add_argument("--local_server_num", type=int, default=2)
parser.add_argument("--http_server_address", type=str, default="127.0.0.1:6666")

args, _ = parser.parse_known_args()
yaml_config = args.yaml_config
tcp_server_ip = args.tcp_server_ip
checkpoint_dir = args.checkpoint_dir
local_server_num = args.local_server_num
http_server_address = args.http_server_address

cur_dir = os.path.dirname(os.path.abspath(__file__))
yaml_config = os.path.join(cur_dir, yaml_config)
checkpoint_dir = os.path.join(cur_dir, checkpoint_dir)

strs = http_server_address.split(":")
http_server_ip = strs[0]
port = strs[1]
assert local_server_num > 0, "The local server number cannot <= 0."

for i in range(local_server_num):
    http_port = int(port) + i
    cmd_server = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && "
    cmd_server += "rm -rf ${execute_path}/logs/server_" + str(http_port) + "/ &&"
    cmd_server += "mkdir -p ${execute_path}/logs/server_" + str(http_port) + "/ &&"
    cmd_server += "cd ${execute_path}/logs/server_" + str(http_port) + "/ || exit && export GLOG_v=1 &&"
    cmd_server += "python ${self_path}/../../run_cloud.py"
    cmd_server += " --ms_role=MS_SERVER"
    cmd_server += " --yaml_config=" + yaml_config
    cmd_server += " --tcp_server_ip=" + tcp_server_ip
    cmd_server += " --checkpoint_dir=" + checkpoint_dir
    cmd_server += " --http_server_address=" + (str(http_server_ip) + ":" + str(http_port))
    cmd_server += " > server.log 2>&1 &"

    import time

    time.sleep(0.3)
    print("subprocess: " + cmd_server)
    subprocess.call(['bash', '-c', cmd_server])
