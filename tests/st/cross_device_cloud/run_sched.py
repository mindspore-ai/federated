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
"""start running lenet scheduler of cross device cloud mode"""

import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run run_cloud.py case")
parser.add_argument("--yaml_config", type=str, default="default_yaml_config.yaml")
parser.add_argument("--scheduler_manage_address", type=str, default="127.0.0.1:11202")

args, _ = parser.parse_known_args()
scheduler_manage_address = args.scheduler_manage_address
yaml_config = args.yaml_config

cur_dir = os.path.dirname(os.path.abspath(__file__))
yaml_config = os.path.join(cur_dir, yaml_config)

cmd_sched = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && rm -rf ${execute_path}/logs/scheduler/ "
cmd_sched += "&& mkdir -p ${execute_path}/logs/scheduler/ &&"
cmd_sched += "cd ${execute_path}/logs/scheduler/ || exit && export GLOG_v=1 &&"
cmd_sched += "python ${self_path}/../../run_cloud.py"
cmd_sched += " --ms_role=MS_SCHED"
cmd_sched += " --yaml_config=" + str(yaml_config)
cmd_sched += " --scheduler_manage_address=" + str(scheduler_manage_address)
cmd_sched += " > scheduler.log 2>&1 &"

print("subprocess: " + cmd_sched)
subprocess.call(['bash', '-c', cmd_sched])
