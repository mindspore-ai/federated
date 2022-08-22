#!/bin/bash
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

# Execute Wide&Deep splitnn demo locally training on criteo dataset. Unlike run_vfl_train_local.sh,
# the embeddings and grad scales are encapsulated using protobuf and are transmitted through socket.

set -e

WORKPATH=$(
  cd "$(dirname $0)" || exit
  pwd
)
export GLOG_v=1
export PYTHONPATH="${PYTHONPATH}:${WORKPATH}/../"

rm -rf follower_socket.log
echo "Start executing Wide&Deep splitnn follower demo."
nohup python run_vfl_train_socket_follower.py >> follower_socket.log  2>&1 &
