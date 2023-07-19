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
"""start run cross silo scheduler"""
import os
import argparse
import subprocess


cur_dir = os.getcwd()
log_dir = './logs/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
code_name = 'test_cross_silo_luna16'

parser = argparse.ArgumentParser(description="Run scheduler case")
parser.add_argument("--yaml_config", type=str, default="default_yaml_config.yaml")
parser.add_argument("--scheduler_manage_address", type=str, default="127.0.0.1:13230")

args, _ = parser.parse_known_args()
scheduler_manage_address = args.scheduler_manage_address
yaml_config = args.yaml_config

yaml_config = os.path.join(cur_dir, yaml_config)

cmd_sched = "rm -rf {}/logs/scheduler/ ".format(cur_dir)
cmd_sched += "&& mkdir {}/logs/scheduler/ &&".format(cur_dir)
cmd_sched += " export GLOG_v=1 &&"
cmd_sched += " python {}/{}.py".format(cur_dir, code_name)
cmd_sched += " --ms_role=MS_SCHED"
cmd_sched += " --yaml_config={}".format(yaml_config)
cmd_sched += " --scheduler_manage_address={}".format(scheduler_manage_address)
cmd_sched += " > {}/logs/scheduler/scheduler.log 2>&1 &".format(cur_dir)

subprocess.call(['bash', '-c', cmd_sched])
