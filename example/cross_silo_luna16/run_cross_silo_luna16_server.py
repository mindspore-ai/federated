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
"""start run cross silo server"""
import os
import argparse
import subprocess


cur_dir = os.getcwd()
log_dir = './logs/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
code_name = 'test_cross_silo_luna16'

parser = argparse.ArgumentParser(description="Run server case")
parser.add_argument("--yaml_config", type=str, default="default_yaml_config.yaml")
parser.add_argument("--tcp_server_ip", type=str, default="127.0.0.1")
parser.add_argument("--checkpoint_dir", type=str, default="./fl_ckpt/")
parser.add_argument("--http_server_address", type=str, default="127.0.0.1:3231")

args, _ = parser.parse_known_args()
yaml_config = args.yaml_config
tcp_server_ip = args.tcp_server_ip
checkpoint_dir = args.checkpoint_dir
http_server_address = args.http_server_address

yaml_config = os.path.join(cur_dir, yaml_config)
checkpoint_dir = os.path.join(cur_dir, checkpoint_dir)

http_port = http_server_address.split(":")[1]

cmd_server = "rm -rf {}/logs/server_{}/ &&".format(cur_dir, http_port)
cmd_server += " mkdir {}/logs/server_{}/ &&".format(cur_dir, http_port)
cmd_server += " export GLOG_v=1 &&"
cmd_server += " python {}/{}.py".format(cur_dir, code_name)
cmd_server += " --ms_role=MS_SERVER"
cmd_server += " --yaml_config={}".format(yaml_config)
cmd_server += " --tcp_server_ip={}".format(tcp_server_ip)
cmd_server += " --checkpoint_dir={}".format(checkpoint_dir)
cmd_server += " --http_server_address={}".format(http_server_address)
cmd_server += " > {}/logs/server_{}/server.log 2>&1 &".format(cur_dir, http_port)

subprocess.call(['bash', '-c', cmd_server])
