#!/bin/bash
set -e
PROJECTPATH=$(cd "$(dirname $0)"; pwd)
export BUILD_PATH="${PROJECTPATH}/build/"

# print usage message
usage()
{
  echo "Usage:"
  echo "    bash build.sh [-j[n]] [-d] [-S on|off]  [-s on|off]"
  echo "    bash build.sh -t on [-j[n]] [-d] [-S on|off]  [-s on|off]"
  echo ""
  echo "Options:"
  echo "    -j[n] Set the threads when building (Default: -j8)"
  echo "    -d Debug model"
  echo "    -t Build testcases."
  echo "    -S Enable enable download cmake compile dependency from gitee instead of github, default off"
  echo "    -s Enable Intel SGX"
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
  # Init default values of build options
  THREAD_NUM=8
  VERBOSE=""
  DEBUG_MODE="off"
  ENABLE_COVERAGE="off"
  ENABLE_ASAN="off"
  ENABLE_PYTHON="on"
  MS_VERSION=""
  RUN_TESTCASES="off"
  ENABLE_GITEE="off"
  ENABLE_SGX="off"

  # Process the options
  while getopts 'dvc:j:a:p:e:V:t:S:s:' opt
  do
    LOW_OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')

    case "${opt}" in
      d)
        echo "user opt: -d"${LOW_OPTARG}
        DEBUG_MODE="on"
        ;;
      j)
        echo "user opt: -j"${LOW_OPTARG}
        THREAD_NUM=$OPTARG
        ;;
      v)
        echo "user opt: -v"${LOW_OPTARG}
        VERBOSE="VERBOSE=1"
        ;;
      c)
        check_on_off $OPTARG c
        ENABLE_COVERAGE="$OPTARG"
        ;;
      a)
        check_on_off $OPTARG a
        ENABLE_ASAN="$OPTARG"
        ;;
      t)
        echo "user opt: -t"${LOW_OPTARG}
        RUN_TESTCASES="$OPTARG"
        ;;
      S)
        check_on_off $OPTARG S
        ENABLE_GITEE="$OPTARG"
        echo "enable download from gitee"
        ;;
      s)
        check_on_off $OPTARG s
        ENABLE_SGX="$OPTARG"
        echo "enable intel sgx"
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}

checkopts "$@"
echo "---------------- MindSpore Federated: build start ----------------"
mkdir -pv "${BUILD_PATH}/package/mindspore_federated/lib"
#if [[ "$MS_BACKEND_HEADER" != "off" ]]; then
#  git submodule update --init third_party/mindspore
#fi

# Create building path
build_mindspore_federated()
{
  echo "start build mindspore_federated project."
  mkdir -pv "${BUILD_PATH}/mindspore_federated"
  cd "${BUILD_PATH}/mindspore_federated"
  CMAKE_ARGS="-DDEBUG_MODE=$DEBUG_MODE -DBUILD_PATH=$BUILD_PATH"
  CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_PYTHON=${ENABLE_PYTHON}"
  CMAKE_ARGS="${CMAKE_ARGS} -DTHREAD_NUM=${THREAD_NUM}"
  if [[ "X$ENABLE_COVERAGE" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_COVERAGE=ON"
  fi
  if [[ "X$ENABLE_ASAN" = "Xon" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_ASAN=ON"
  fi
  if [[ "$MS_VERSION" != "" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DMS_VERSION=${MS_VERSION}"
  fi
  if [[ "X$RUN_TESTCASES" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_TESTCASES=ON"
  fi
  if [[ "X$ENABLE_GITEE" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GITEE=ON"
  fi
  if [[ "X$ENABLE_SGX" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_SGX=ON"
  fi
  echo "${CMAKE_ARGS}"
  cmake ${CMAKE_ARGS} ../..
  if [[ -n "$VERBOSE" ]]; then
    CMAKE_VERBOSE="--verbose"
  fi
  cmake --build . --target package ${CMAKE_VERBOSE} -j$THREAD_NUM
  echo "success building mindspore_federated project!"
}

build_mindspore_federated

echo "---------------- mindspore_federated: build end   ----------------"
