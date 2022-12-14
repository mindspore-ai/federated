# Copyright 2022 Huawei Technologies Co., Ltd
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
"""stop client"""

import argparse
import os

parser = argparse.ArgumentParser(description="Finish FLClient case")
parser.add_argument("--kill_tag", type=str, default="mindspore-lite-java-flclient")

args, _ = parser.parse_known_args()
kill_tag = args.kill_tag

cmd = "pid=`ps -ef|grep " + kill_tag
cmd += " |grep -v \"grep\" | grep -v \"finish\" |awk '{print $2}'` && "
cmd += "for id in $pid; do kill -9 $id && echo \"killed $id\"; done"

os.system(cmd)
