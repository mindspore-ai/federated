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
"""Run data join."""
import os
from mindspore_federated import FLDataWorker
from mindspore_federated.common.config import get_config


def mkdir(directory):
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass


if __name__ == '__main__':
    mkdir("vfl")
    mkdir("vfl/output")
    mkdir("vfl/output/leader")
    mkdir("vfl/output/follower")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    args = get_config(os.path.join(current_dir, "vfl/vfl_data_join_config.yaml"))
    dict_cfg = args.__dict__

    worker = FLDataWorker(config=dict_cfg)
    worker.do_worker()
