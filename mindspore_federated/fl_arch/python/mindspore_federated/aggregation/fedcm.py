# Copyright 2023 Huawei Technologies Co., Ltd
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
"""loss cell definition of fedcm model"""
# from mindspore import load_param_into_net
import numpy as np
from mindspore.nn import Cell
from mindspore.ops import Concat, ReduceSum, Mul

def get_mdl_params(model_list, n_par=None):
    """get function parameters"""
    if n_par is None:
        exp_mdl = model_list[0]
        n_par = 0
        for param in exp_mdl.trainable_params():
            n_par += len(param.asnumpy().reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype("float32")
    for i, mdl in enumerate(model_list):
        idx = 0
        for param in mdl.trainable_params():
            temp = param.asnumpy().reshape(-1)
            param_mat[i, idx : idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

def extend_model(networkn):
    """extend model"""
    extenednet = networkn
    local_par_lists = None
    for param in extenednet.trainable_params():
        if local_par_lists is None:
            local_par_lists = param.reshape(-1)
        else:
            local_par_lists = Concat()((local_par_lists, param.reshape(-1)))
    return local_par_lists

class FedCMWithLossCell(Cell):
    """define fedcm loss"""
    def __init__(self, backbone, local_par_list, loss_fn, delta, loss_callback, preloss):
        """construct function for fedcm cell"""
        super(FedCMWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._delta = delta
        self._local_par_list = local_par_list
        self._losscallback = loss_callback
        self._preloss = Parameter(preloss)

    def get_preloss(self):
        """get proloss"""
        return self._preloss

    def construct(self, data, label):
        """construct the loss function"""
        alpha = 0.02
        out = self._backbone(data)

        loss = self._loss_fn(out, label)
        loss_algo = ReduceSum()(Mul()(self._local_par_list, self._delta))

        self._preloss = loss
        loss = alpha * loss + (alpha - 1) * loss_algo
        return loss

    @property
    def backbone_network(self):
        """backbone network"""
        return self._backbone
