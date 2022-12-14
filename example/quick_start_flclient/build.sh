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

display_usage()
{
    echo -e "Usage:"
    echo "bash build.sh -r $FL_CLIENT_PATH"
    echo "Options:"
    echo "    -r Absolute path of mindspore-lite-java-flclient.jar"
}

checkopts()
{
  FL_CLIENT_PATH=""
  while getopts 'r:' opt
  do
    case "${opt}" in
      r)
        FL_CLIENT_PATH=$OPTARG
        ;;
      *)
        echo "Unknown option ${opt}!"
        display_usage
        exit 1
    esac
  done
}

checkopts "$@"

if [ ! -e $FL_CLIENT_PATH ] ; then
  echo "$FL_CLIENT_PATH not exist, please check you input path"
  exit
fi

BASEPATH=$(cd "$(dirname $0)" || exit; pwd)
rm -rf lib
mkdir -p lib

cp $FL_CLIENT_PATH ${BASEPATH}/lib/

mvn package -Dmaven.test.skip=true
