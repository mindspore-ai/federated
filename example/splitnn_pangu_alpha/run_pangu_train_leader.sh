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

# Execute pangu-splitnn distributed training demo on wiki dataset.
export CUDA_VISIBLE_DEVICES=0
set -e

WORKPATH=$(
  cd "$(dirname $0)" || exit
  pwd
)

HTTP_SERVER_ADDRESS=$1
REMOTE_SERVER_ADDRESS=$2
DATA_URL=$3
EVAL_DATA_URL=$4
ENABLE_SSL=$5
SERVER_PASSWORD=$6
CLIENT_PASSWORD=$7
SERVER_CERT_PATH=$8
CLIENT_CERT_PATH=$9
CA_CERT_PATH=${10}
EMBEDDING_DP=${11}

export GLOG_v=1
export PYTHONPATH="${PYTHONPATH}:${WORKPATH}/../"
pid=`ps -ef|grep http_server_address=$HTTP_SERVER_ADDRESS |grep -v "grep" |awk '{print $2}'` && for id in $pid; do kill -9 $id && echo "killed $id"; done

echo "run_pangu_train_leader.py is started."
rm -rf $WORKPATH/run_pangu_train_leader.log
nohup python run_pangu_train_leader.py \
  --http_server_address=$HTTP_SERVER_ADDRESS \
  --remote_server_address=$REMOTE_SERVER_ADDRESS  \
  --data_url=$DATA_URL \
  --eval_data_url=$EVAL_DATA_URL \
  --enable_ssl=$ENABLE_SSL \
  --server_password=$SERVER_PASSWORD \
  --client_password=$CLIENT_PASSWORD \
  --client_cert_path=$CLIENT_CERT_PATH\
  --server_cert_path=$SERVER_CERT_PATH \
  --ca_cert_path=$CA_CERT_PATH \
  --embedding_dp=$EMBEDDING_DP >> $WORKPATH/run_pangu_train_leader.log 2>&1 &
