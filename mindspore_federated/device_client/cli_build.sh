#!/bin/bash
set -e
PROJECT_PATH=$(cd "$(dirname "$0")"; pwd)
# Init default values of build options
FL_THIRD_PKG_PATH="${PROJECT_PATH}/third/"
USE_CACHED_PKG="on"
ENABLE_GITEE="on"
MS_LITE_PKG_VER="1.9.0"
MS_LITE_PKG_NAME="mindspore-lite-${MS_LITE_PKG_VER}-linux-x64"
MS_PKG_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_LITE_PKG_VER}/MindSpore/lite/release/linux/x86_64/${MS_LITE_PKG_NAME}.tar.gz"
LENET_MS_URL="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/models/lenet_train.ms"

# print usage message
usage()
{
  echo "Usage:"
  echo "    bash cli_build.sh [-p path] [-c on|off] [-S on|off] [-h] "
  echo ""
  echo "Options:"
  echo "    -p set the path of mindspore lite package"
  echo "    -c using the cached mindspore lite package or not, while 'on' check package exist or not before auto download"
  echo "    -S download third party package from gitee"
  echo "    -h print usage info"
}

# check value of input is 'on' or 'off'
# usage: check_on_off arg_value arg_name
check_on_off()
{
  if [[ "X$1" != "Xon" && "X$1" != "Xoff" ]]; then
    echo "Invalid value $1 for option -$2"
    usage
    exit 1
  fi
}

# check and set options
checkopts()
{
  # Process the options
  while getopts 'p:c:h:S' opt
  do
    LOW_OPT_ARG=$(echo "${OPTARG}" | tr '[:upper:]' '[:lower:]')

    case "${opt}" in
      p)
        echo "user opt: -p ""${OPTARG}"
        FL_THIRD_PKG_PATH=${OPTARG}
        ;;
      c)
        echo "user opt: -c ""${OPTARG}"
        USE_CACHED_PKG=${LOW_OPT_ARG}
        ;;
      S)
        echo "user opt: -c ""${OPTARG}"
        ENABLE_GITEE=${LOW_OPT_ARG}
        ;;
      h)
        echo "user opt: -h"
        usage
        exit 1
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}

load_ms_lite_pkg(){
  # load mindspore lite pkg
  cd "$FL_THIRD_PKG_PATH"
  echo "start load ${MS_LITE_PKG_NAME}  ..."
  echo "The lite pkg save path is:${FL_THIRD_PKG_PATH}"
  if [ "X$USE_CACHED_PKG" == "Xon" ] && [ -e "${MS_LITE_PKG_NAME}.tar.gz" ]; then
    echo "Find cached lite pkg, don't load again"
    return
  fi

  rm -f "$FL_THIRD_PKG_PATH"/${MS_LITE_PKG_NAME}.tar.gz
  rm -rf "$FL_THIRD_PKG_PATH"/${MS_LITE_PKG_NAME:?}
  wget --no-check-certificate ${MS_PKG_URL}
  if [ ! -e "${MS_LITE_PKG_NAME}.tar.gz" ] ; then
    echo "down load ${MS_LITE_PKG_NAME} failed, please download manually or check your net config ..."
    exit
  fi
  echo "load ${MS_LITE_PKG_NAME} success ..."
}

load_flat_buffer_pkg(){
  # load flat buffer
  cd "$FL_THIRD_PKG_PATH"
  if [ "X$USE_CACHED_PKG" == "Xon" ] && [ -e "flatbuffers-v2.0.0/build/flatc" ]; then
    echo "Find cached flat pkg, don't load again"
    return
  fi
  rm -f "$FL_THIRD_PKG_PATH"/v2.0.0.tar.gz
  rm -rf "$FL_THIRD_PKG_PATH"/flatbuffers-v2.0.0
  if [ "X$ENABLE_GITEE" == "Xon" ]; then
     wget --no-check-certificate https://gitee.com/mirrors/flatbuffers/repository/archive/v2.0.0.tar.gz
     tar -zxf v2.0.0.tar.gz
  else
     wget --no-check-certificate https://github.com/google/flatbuffers/archive/v2.0.0.tar.gz
     tar -zxf v2.0.0.tar.gz
     mv "$FL_THIRD_PKG_PATH"/flatbuffers-2.0.0 "$FL_THIRD_PKG_PATH"/flatbuffers-v2.0.0
  fi
  mkdir -p "$FL_THIRD_PKG_PATH"/flatbuffers-v2.0.0/build
  cd "$FL_THIRD_PKG_PATH"/flatbuffers-v2.0.0/build
  cmake ..
  make -j4
  if [ ! -e "flatc" ]; then
    echo "down load or compile flatc failed, please do manually ..."
    exit
  fi
  echo "load and compile flat buffer success ..."
}


load_third_pkg()
{
  mkdir -p "$FL_THIRD_PKG_PATH"
  load_ms_lite_pkg
  load_flat_buffer_pkg
}

checkopts "$@"
export FLAT_EXE_PATH="$FL_THIRD_PKG_PATH/flatbuffers-v2.0.0/build/flatc"
echo "---------------- MindSpore Federated client: build start ----------------"
load_third_pkg

# Create building path
build_mindspore_federated_client()
{
  echo "start build mindspore_federated client project."
  mkdir -p "${PROJECT_PATH}"/libs/
  rm -rf "${PROJECT_PATH}"/libs/*
  tar -zxf "${FL_THIRD_PKG_PATH}"/${MS_LITE_PKG_NAME}.tar.gz -C "${FL_THIRD_PKG_PATH}"/
  cp "${FL_THIRD_PKG_PATH}"/${MS_LITE_PKG_NAME}/runtime/lib/mindspore-lite-java.jar "${PROJECT_PATH}"/libs
  cp "${FL_THIRD_PKG_PATH}"/${MS_LITE_PKG_NAME}/runtime/third_party/libjpeg-turbo/lib/* "${FL_THIRD_PKG_PATH}"/${MS_LITE_PKG_NAME}/runtime/lib/
  cd "${PROJECT_PATH}"

  rm -rf gradle .gradle gradlew gradlew.bat
  local gradle_version=""
  gradle_version=$(gradle --version | grep Gradle | awk '{print$2}')
  if [[ ${gradle_version} == '6.6.1' ]]; then
    gradle_command=gradle
  else
    gradle wrapper --gradle-version 6.6.1 --distribution-type all
    gradle_command="${PROJECT_PATH}"/gradlew
  fi

  ${gradle_command} clean
  ${gradle_command} createFlatBuffers
  # compile flclient
  ${gradle_command} build -x test
  ${gradle_command} packFLJarAAR --rerun-tasks
  ${gradle_command} packFLJarX86 --rerun-tasks
  ${gradle_command} createPom --rerun-tasks

  # compile ut depend jar
  cd ${PROJECT_PATH}/../../example/quick_start_flclient/
  bash build.sh
  cp ${PROJECT_PATH}/../../example/quick_start_flclient/build/libs/quick_start_flclient.jar ${PROJECT_PATH}/libs/
  cd "${PROJECT_PATH}"

  # do ut test
  ${gradle_command} packFLJarX86UT --rerun-tasks
  export LD_LIBRARY_PATH=${FL_THIRD_PKG_PATH}/${MS_LITE_PKG_NAME}/runtime/lib/:${LD_LIBRARY_PATH}
  export MS_FL_UT_BASE_PATH=${PROJECT_PATH}/ut_data
  mkdir -p ${PROJECT_PATH}/ut_data/test_data/lenet/f0178_39/
  cd ${PROJECT_PATH}/ut_data
  # the ci build server can't run gen_lenet_data
  # python gen_lenet_data.py
  cd ${PROJECT_PATH}/
  wget --no-check-certificate ${LENET_MS_URL} -P ${PROJECT_PATH}/ut_data/test_data/lenet/
  ${gradle_command} test
}

java -version
echo "JAVA_HOM is ${JAVA_HOME}"

build_mindspore_federated_client

echo "---------------- mindspore_federated client: build end   ----------------"
