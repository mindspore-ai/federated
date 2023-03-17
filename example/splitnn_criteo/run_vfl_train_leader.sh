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
ENABLE_SSL=$4
SERVER_PASSWORD=$5
CLIENT_PASSWORD=$6
SERVER_CERT_PATH=$7
CLIENT_CERT_PATH=$8
CA_CERT_PATH=$9

export GLOG_v=1
export PYTHONPATH="${PYTHONPATH}:${WORKPATH}/../"

pid=`ps -ef|grep http_server_address=$HTTP_SERVER_ADDRESS |grep -v "grep" |awk '{print $2}'` && for id in $pid; do kill -9 $id && echo "killed $id"; done

echo "run_vfl_train_leader.py is started."
rm -rf $WORKPATH/vfl_train_leader.log
nohup python run_vfl_train_leader.py \
  --data_path=$DATA_PATH \
  --http_server_address=$HTTP_SERVER_ADDRESS \
  --remote_server_address=$REMOTE_SERVER_ADDRESS \
  --enable_ssl=$ENABLE_SSL \
  --server_password=$SERVER_PASSWORD \
  --client_password=$CLIENT_PASSWORD \
  --client_cert_path=$CLIENT_CERT_PATH\
  --server_cert_path=$SERVER_CERT_PATH \
  --ca_cert_path=$CA_CERT_PATH >> $WORKPATH/vfl_train_leader.log 2>&1 &
