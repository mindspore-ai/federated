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

BASEPATH=$(cd "$(dirname $0)" || exit; pwd)

FL_THIRD_PKG_PATH="${BASEPATH}/../../mindspore_federated/device_client/third/"
MS_LITE_PKG_VER="1.9.0"
MS_LITE_PKG_NAME="mindspore-lite-${MS_LITE_PKG_VER}-linux-x64"
MS_PKG_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_LITE_PKG_VER}/MindSpore/lite/release/linux/x86_64/${MS_LITE_PKG_NAME}.tar.gz"
USE_CACHED_PKG="on"
display_usage()
{
    echo -e "Usage:"
    echo "bash build.sh -r $FL_CLIENT_JAR_PATH [-c on|off]"
    echo "Options:"
    echo "    -r Absolute path of mindspore-lite-java-flclient.jar"
    echo "    -c using the cached mindspore lite package or not, while 'on' check package exist or not before auto download"
}

checkopts()
{
  FL_CLIENT_PATH=$(ls ${BASEPATH}/../../mindspore_federated/device_client/build/libs/jarX86/mindspore-lite-java-flclient-*jar)
  while getopts 'r:c:' opt
  do
    case "${opt}" in
      r)
        FL_CLIENT_PATH=$OPTARG
        ;;
      c)
        echo "user opt: -c ""${OPTARG}"
        USE_CACHED_PKG=${LOW_OPT_ARG}
        ;;
      *)
        echo "Unknown option ${opt}!"
        display_usage
        exit 1
    esac
  done
}

load_ms_lite_pkg(){
  # load mindspore lite pkg
  mkdir -p "$FL_THIRD_PKG_PATH"
  echo "start load ${MS_LITE_PKG_NAME}  ..."
  echo "The lite pkg save path is:${FL_THIRD_PKG_PATH}"
  if [ "X$USE_CACHED_PKG" == "Xon" ] && [ -e "${FL_THIRD_PKG_PATH}/${MS_LITE_PKG_NAME}.tar.gz" ]; then
    echo "Find cached lite pkg, don't load again"
    return
  fi

  rm -f "$FL_THIRD_PKG_PATH"/${MS_LITE_PKG_NAME}.tar.gz
  rm -rf "$FL_THIRD_PKG_PATH"/${MS_LITE_PKG_NAME:?}
  wget --no-check-certificate  -P "$FL_THIRD_PKG_PATH" ${MS_PKG_URL}
  if [ ! -e "${FL_THIRD_PKG_PATH}/${MS_LITE_PKG_NAME}.tar.gz" ] ; then
    echo "down load ${FL_THIRD_PKG_PATH}/${MS_LITE_PKG_NAME} failed, please download manually or check your net config
    ..."
    exit
  fi
  echo "load ${MS_LITE_PKG_NAME} success ..."
}


checkopts "$@"

if [ ! -e $FL_CLIENT_PATH ] ; then
  echo "$FL_CLIENT_PATH not exist, please check you input path"
  exit
fi

rm -rf lib
mkdir -p lib

load_ms_lite_pkg

# copy dependency jar to lib
tar -zxf "${FL_THIRD_PKG_PATH}"/${MS_LITE_PKG_NAME}.tar.gz -C "${FL_THIRD_PKG_PATH}"/
cp ${FL_THIRD_PKG_PATH}/${MS_LITE_PKG_NAME}/runtime/lib/mindspore-lite-java.jar ${BASEPATH}/lib/

cp $FL_CLIENT_PATH ${BASEPATH}/lib/

gradle_version=$(gradle --version | grep Gradle | awk '{print$2}')
if [[ ${gradle_version} == '6.6.1' ]]; then
  gradle_command=gradle
else
  gradle wrapper --gradle-version 6.6.1 --distribution-type all
  gradle_command="${PROJECT_PATH}"/gradlew
fi
${gradle_command} clean
${gradle_command} build
