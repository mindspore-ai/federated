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
"""train_criteo."""
import os
import sys
from mindspore import context
from mindspore.common import set_seed
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_num

from server.fedavg import FedAvg
from server.fedasync import FedAsync

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config.rank_size = get_device_num()

set_seed(1)
def modelarts_pre_process():
    pass

@moxing_wrapper(pre_process=modelarts_pre_process)
def train_deepfm():

    if config.device_target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
        config.rank_size = 1
        rank_id = None
        # context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
        config.rank_size = None
        rank_id = None
    #设置fl模式
    fl_server = {
        "fedavg": FedAvg(config),
        "fedasync": FedAsync(config)
    }[config.fl_mode]
    fl_server.boot()
    fl_server.run()


if __name__ == '__main__':
    train_deepfm()