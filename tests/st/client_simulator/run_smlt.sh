#!/bin/bash
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

export PYTHONPATH=../../../../:$PYTHONPATH
client_num=$1
http_type=$2
http_server_address=$3
checkpoint_path=$4

for((i=0;i<client_num;i++));
do
  echo $http_server_address
  python simulator.py --pid=$i --http_type=$http_type --http_server_address=$http_server_address --checkpoint_path=$checkpoint_path> simulator_$i.log 2>&1 &
done
