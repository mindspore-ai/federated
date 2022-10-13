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

set -e

usage()
{
  echo "Usage:"
  echo "bash run_benchmark_net.sh [-h] [-l level] [-r PATH] [-p PATH] [-j PATH] [-m PATH]"
  echo "Options:"
  echo "    -h Print usage"
  echo "    -l st level, now not used"
  echo "    -r fl resource path"
  echo "    -p mindspore x86_whl and lite_x86 pkg/"
  echo "    -j java jdk path, version 1.9 or later"
}

#base_path=$(dirname "$(readlink -f "$0")")
# Process the options
while getopts 'h:l:r:p:j:' opt
do
  case "${opt}" in
    h)
      usage;;
    l)
      FL_TEST_LEVEL=${OPTARG}
      echo "FL_TEST_LEVEL is ${FL_TEST_LEVEL}";;
    r)
      FL_RESOURCE_PATH=${OPTARG}
      echo "FL_RESOURCE_PATH is ${FL_RESOURCE_PATH}";;
    p)
      X86_PKG_PATH=${OPTARG}
      echo "X86_PKG_PATH is ${X86_PKG_PATH}";;
    j)
      FL_JDK_PATH=${OPTARG}
      echo "FL_JDK_PATH is ${FL_JDK_PATH}";;
    *)
      echo "Unknown option ${opt}!"
      usage
      exit 1
  esac
done

if [ "X${FL_JDK_PATH}" == "X" ] && [ "X${JAVA_HOME}" != "X" ]; then
  FL_JDK_PATH=${JAVA_HOME}/bin
fi

if [ "X${FL_RESOURCE_PATH}" == "X" ] || [ "X${X86_PKG_PATH}" == "X" ] ||  \
  [ "X${FL_JDK_PATH}" == "X" ]; then
  echo "FL_RESOURCE_PATH, X86_PKG_PATH, FL_JDK_PATH must be set."
  usage
  exit 1
fi

# get resources for v1.8.0
# to support rollback we provide multi version of data
#mkdir -p ${base_path}/fl_resources
#if [ ! -f ${FL_RESOURCE_PATH}/v1.8.0_encrypt_st.tar ]; then
#   echo "${FL_RESOURCE_PATH}/v1.8.0_encrypt_st not exist."
#   exit 1
#fi
#tar -xf ${FL_RESOURCE_PATH}/v1.8.0_encrypt_st.tar -C ${base_path}/fl_resources

export FL_RESOURCE_PATH=${FL_RESOURCE_PATH}
export X86_PKG_PATH=${X86_PKG_PATH}
export FL_JDK_PATH=${FL_JDK_PATH}
export MS_INSTAll_PKG=$(ls ${X86_PKG_PATH}/mindspore-[0-9].[0-9].[0-9]*-cp37-cp37m-linux_x86_64.whl)
export MS_LITE_PKG=$(ls ${X86_PKG_PATH}/mindspore-lite-*-linux-x64.tar.gz)
export FL_INSTALL_PKG=$(ls ${X86_PKG_PATH}/mindspore_federated-*-cp37-cp37m-linux_x86_64.whl)
export FL_ClI_FRAME_JAR=$(ls ${X86_PKG_PATH}/mindspore-lite-java-flclient-*.jar)

if [ "X${MS_INSTAll_PKG}" == "X" ]  || [ ! -f ${MS_INSTAll_PKG} ]  || \
   [ "X${MS_LITE_PKG}" == "X" ] || [ ! -f ${MS_LITE_PKG} ] ||    \
   [ "X${FL_INSTALL_PKG}" == "X" ] || [ ! -f ${FL_INSTALL_PKG} ] || \
   [ "X${FL_ClI_FRAME_JAR}" == "X" ]|| [ ! -f ${FL_ClI_FRAME_JAR} ]; then
  echo "MindSpore pkg, lite pkg, FL pkg must be provided in ${X86_PKG_PATH}."
  usage
  exit 1
fi

pytest -s -v st_script
#pytest -s -v st_script/test_vae.py
#pytest -s -v st_script/test_adstag.py::TestAdsTagTrain::test_train_adstag_nc_ne
#pytest -s -v st_script/test_albert.py::TestAlbertTrain::test_train_albert_nc_ne
#pytest -s -v st_script/test_lenet.py::TestLenetTrain::test_train_lenet_batchsize_16
#pytest -s -v st_script/test_lenet.py::TestLenetTrain::test_train_lenet_nc_ne
ret=$?
exit $ret