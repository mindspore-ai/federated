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

# Execute Wide&Deep splitnn demo locally training on criteo dataset.
export CUDA_VISIBLE_DEVICES=0
set -e

WORKPATH=$(
  cd "$(dirname $0)" || exit
  pwd
)

vocab_folder="./bpe_4w_pcl"
if [ ! -d "$vocab_folder" ]; then
  mkdir "$vocab_folder"
fi
vocab_model="./bpe_4w_pcl/vocab.model"
if [ ! -f "$vocab_model" ]; then
  wget -P ./bpe_4w_pcl https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/tokenizer/vocab.model --no-check-certificate
fi

export PYTHONPATH="${PYTHONPATH}:${WORKPATH}/../"
echo "Start executing pangu_alpha splitnn demo (standalone simulation mode)."
python run_pangu_train_local.py
