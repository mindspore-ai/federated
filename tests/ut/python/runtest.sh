#!/bin/bash
# Copyright 2019 Huawei Technologies Co., Ltd
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
set -e
BASEPATH=$(
  cd "$(dirname "$0")"
  pwd
)
ROOT_DIR=${BASEPATH}/../../..

cp -r ${ROOT_DIR}/mindspore_federated/fl_arch/python/mindspore_federated/* ${ROOT_DIR}/build/package/mindspore_federated/

export PYTHONPATH=${ROOT_DIR}/build/package:${ROOT_DIR}/tests/ut/python/tests:$PYTHONPATH

export LD_LIBRARY_PATH=${ROOT_DIR}/build/package/mindspore_federated:${ROOT_DIR}/build/package/mindspore_federated/lib:${LD_LIBRARY_PATH}

echo "PYTHONPATH=$PYTHONPATH"
export GLOG_v=1
export REDIS_SERVER_PORT=12345

unset http_proxy
unset https_proxy

function clear_port()
{
  for port in [ 3001 3002 3003 3004 ]
  do
    PROCESS=$(netstat -nlp | grep :$port | awk '{print $7}' | awk -F"/" '{print $1}')
    for i in $PROCESS
       do
       echo "Kill the process [ $i ]"
       kill -9 $i
    done
  done
}

function start_redis_server() {
  echo "begin start redis server"
  redis-server --port ${REDIS_SERVER_PORT} --save "" &
  sleep 0.5s
  echo "after start redis server"
  ps aux | grep 'redis-server' | grep :${REDIS_SERVER_PORT} | grep -v grep

  count=$(ps aux | grep 'redis-server' | grep :${REDIS_SERVER_PORT} | grep -v grep | wc -l)
  if [[ $count == 0 ]]
  then
    echo "Failed to start redis server, server port: ${REDIS_SERVER_PORT}"
    exit 1
  fi
  echo "end start redis server"
}

function stop_redis_server()
{
    echo "begin stop redis server"
    pid=$(ps aux | grep 'redis-server' | grep :${REDIS_SERVER_PORT} | grep -v grep | awk '{print $2}')
    for id in $pid; do kill -15 $id; done
    sleep 0.5s
    pid=$(ps aux | grep 'redis-server' | grep :${REDIS_SERVER_PORT} | grep -v grep | awk '{print $2}')
    for id in $pid; do kill -9 $id; done
    echo "after stop redis server"
    ps aux | grep 'redis-server'
}

clear_port

cd ${ROOT_DIR}/tests/ut/python/tests/

bash ${BASEPATH}/generate_certs.sh
echo "ls fl_ssl_cert:"
ls ${BASEPATH}/fl_ssl_cert

bash ${BASEPATH}/generate_certs_redis.sh
echo "ls fl_redis_ssl_cert:"
ls ${BASEPATH}/fl_redis_ssl_cert

stop_redis_server
start_redis_server
if [ $# -gt 0 ]; then
  pytest -s -v . -k "$1"
else
  pytest -v .
fi
tests_result=$?
stop_redis_server

exit $tests_result
