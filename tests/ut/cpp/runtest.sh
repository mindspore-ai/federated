#!/bin/bash

set -e
BASEPATH=$(cd "$(dirname "$0")"; pwd)
PROJECT_PATH=${BASEPATH}/../../..
if [ $BUILD_PATH ];then
    echo "BUILD_PATH = $BUILD_PATH"
else
    BUILD_PATH=${PROJECT_PATH}/build
    echo "BUILD_PATH = $BUILD_PATH"
fi
cd ${BUILD_PATH}/mindspore_federated/tests/ut/cpp


export LD_LIBRARY_PATH=${BUILD_PATH}/mindspore_federated/googletest/googlemock/gtest:\
${PROJECT_PATH}/mindspore_federated/python/mindspore_federated:\
${PROJECT_PATH}/mindspore_federated/python/mindspore_federated/lib:\
${PROJECT_PATH}/graphengine/third_party/prebuild/x86_64:\
${PROJECT_PATH}/graphengine/third_party/prebuild/aarch64:${LD_LIBRARY_PATH}
export PYTHONPATH=${PROJECT_PATH}/tests/ut/cpp/python_input:\
$PYTHONPATH:${PROJECT_PATH}/mindspore_federated/python:\
${PROJECT_PATH}/tests/ut/python:${PROJECT_PATH}
export GLOG_v=1
export GC_COLLECT_IN_CELL=1
## set op info config path
export MINDSPORE_OP_INFO_PATH=${PROJECT_PATH}/config/op_info.config

## prepare data for dataset & mindrecord
#cp -fr $PROJECT_PATH/tests/ut/data ${PROJECT_PATH}/build/mindspore_federated/tests/ut/cpp/
## prepare album dataset, uses absolute path so has to be generated
#python ${PROJECT_PATH}/build/mindspore_federated/tests/ut/cpp/data/dataset/testAlbum/gen_json.py

if [ $# -gt 0 ]; then
  ./ut_tests --gtest_filter=$1
else
  ./ut_tests
fi
RET=$?
cd -

exit ${RET}
