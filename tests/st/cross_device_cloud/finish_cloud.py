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
"""Cross device cloud finish."""

import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Finish cross_device_cloud case")
parser.add_argument("--redis_port", type=int, default=8113)

args, _ = parser.parse_known_args()
redis_port = args.redis_port

cur_dir = os.path.dirname(os.path.abspath(__file__))

cmd = "pid=`ps -ef|grep \"" + str(cur_dir) + "\" "
cmd += " | grep -v \"grep\" | grep -v \"finish\" |awk '{print $2}'` && "
cmd += "for id in $pid; do kill -9 $id && echo \"killed $id\"; done"

subprocess.call(['bash', '-c', cmd])


cmd = "pid=`ps -ef|grep redis | grep " + str(redis_port)
cmd += " | grep -v \"grep\" | grep -v \"finish\" |awk '{print $2}'` && "
cmd += "for id in $pid; do kill -9 $id && echo \"killed $id\"; done"

subprocess.call(['bash', '-c', cmd])
