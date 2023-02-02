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

# Execute Wide&Deep split nn demo leader training on Criteo dataset with type of MindRecord.
# The embeddings and grad scales are transmitted through http.

set -e

WORKPATH=$(
  cd "$(dirname $0)" || exit
  pwd
)
HTTP_SERVER_ADDRESS=$1
REMOTE_SERVER_ADDRESS=$2
DATA_PATH=$3
RESUME=$4
COMPRESS=$5

export GLOG_v=1
export PYTHONPATH="${PYTHONPATH}:${WORKPATH}/../"

pid=`ps -ef|grep http_server_address=$HTTP_SERVER_ADDRESS |grep -v "grep" |awk '{print $2}'` && for id in $pid; do kill -9 $id && echo "killed $id"; done

echo "run_vfl_train_leader.py is started."
rm -rf $WORKPATH/vfl_train_leader.log
nohup python run_vfl_train_leader.py --label_dp=True \
  --data_path=$DATA_PATH \
  --resume=$RESUME \
  --http_server_address=$HTTP_SERVER_ADDRESS \
  --remote_server_address=$REMOTE_SERVER_ADDRESS \
  --compress=$COMPRESS >> $WORKPATH/vfl_train_leader.log 2>&1 &
