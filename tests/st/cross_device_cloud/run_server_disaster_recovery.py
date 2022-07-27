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

# The script runs the process of server's disaster recovery. It will kill the server process and launch it again.
"""start running lenet recovery server of cross device cloud mode"""

import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description="Run run_cloud.py case")

parser.add_argument("--yaml_config", type=str, default="default_yaml_config.yaml")
parser.add_argument("--tcp_server_ip", type=str, default="127.0.0.1")
parser.add_argument("--checkpoint_dir", type=str, default="./fl_ckpt/")
parser.add_argument("--fl_server_port", type=int, default=6666)

args, _ = parser.parse_known_args()
yaml_config = args.yaml_config
tcp_server_ip = args.tcp_server_ip
checkpoint_dir = args.checkpoint_dir
http_port = args.fl_server_port

cur_dir = os.path.dirname(os.path.abspath(__file__))
yaml_config = os.path.join(cur_dir, yaml_config)
checkpoint_dir = os.path.join(cur_dir, checkpoint_dir)

http_param = "http_server_address=127.0.0.1:" + str(http_port)

# Step 1: make the server offline.
offline_cmd = "ps_demo_id=`ps -ef | grep " + http_param \
              + "|grep -v cd  | grep -v grep | grep -v run_server_disaster_recovery | awk '{print $2}'`"
offline_cmd += " && for id in $ps_demo_id; do kill -9 $id && echo \"Killed server process: $id\"; done"
subprocess.call(['bash', '-c', offline_cmd])

# Step 2: Wait 3 seconds for recovery.
wait_cmd = "echo \"Start to sleep for 3 seconds\" && sleep 3"
subprocess.call(['bash', '-c', wait_cmd])

# Step 3: Launch the server again with the same fl server port.
cmd_server = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && "
cmd_server += "rm -rf ${execute_path}/logs/disaster_recovery_server_" + str(http_port) + "/ &&"
cmd_server += "mkdir -p ${execute_path}/logs/disaster_recovery_server_" + str(http_port) + "/ &&"
cmd_server += "cd ${execute_path}/logs/disaster_recovery_server_" + str(http_port) + "/ || exit && export GLOG_v=1 &&"
cmd_server += "python ${self_path}/../../run_cloud.py"
cmd_server += " --ms_role=MS_SERVER"
cmd_server += " --yaml_config=" + yaml_config
cmd_server += " --tcp_server_ip=" + tcp_server_ip
cmd_server += " --checkpoint_dir=" + checkpoint_dir
cmd_server += " --http_server_address=127.0.0.1:" + str(http_port)
cmd_server += " > server.log 2>&1 &"

subprocess.call(['bash', '-c', cmd_server])
