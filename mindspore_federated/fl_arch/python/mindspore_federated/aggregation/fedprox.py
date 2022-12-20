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
"""HFL_FedProx_Cell_wrapper."""

from __future__ import absolute_import
from __future__ import division
from mindspore import ops
from mindspore.nn.cell import Cell
from mindspore.common.parameter import ParameterTuple


class WithFedProxLossCell(Cell):
    r"""
    Cell with loss function for FedProx aggregation.

    Wraps the network with loss function. This Cell accepts data and label and global parameter tuple as inputs and
    the computed loss will be returned.

    Args:
        backbone (Cell): The backbone network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
        iid_rate (float): The ratio of difference between local weights and global weights in loss function

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a tensor means the loss value, the shape of which is usually :math:`()`.

    Raises:
        TypeError: If dtype of `data` or `label` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
        >>> net_with_criterion = nn.WithFedProxLossCell(net, loss_fn, iid_rate=0.1)
        >>>
        >>> batch_size = 2
        >>> data = Tensor(np.ones([batch_size, 1, 32, 32]).astype(np.float32) * 0.01)
        >>> label = Tensor(np.ones([batch_size, 10]).astype(np.float32))
        >>>
        >>> output_data = net_with_criterion(data, label)
    """

    def __init__(self, backbone, loss_fn, iid_rate):
        super(WithFedProxLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self.weights = ParameterTuple(self.filter_not_contain_names("global_weights"))
        self.last_global_weights = ParameterTuple(self.filter_contain_names("global_weights"))
        self._loss_fn = loss_fn
        self.iid_rate = iid_rate
        self.len = len(self.weights)
        self.reducemean = ops.ReduceMean()

    def construct(self, data, label):
        out = self._backbone(data)
        fed_prox_reg = 0.0
        for i in range(self.len):
            diff = self.reducemean((self.weights[i] - self.last_global_weights[i]) ** 2)
            fed_prox_reg = fed_prox_reg + (self.iid_rate / 2.0) * diff
        return self._loss_fn(out, label) + fed_prox_reg

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, the backbone network.
        """
        return self._backbone

    def filter_contain_names(self, name, recurse=True):
        """
        Check the names of cell parameters.
        """
        return list(filter(lambda x: name in x.name, self.get_parameters(expand=recurse)))

    def filter_not_contain_names(self, name):
        """
        Check the names of cell parameters.
        """
        return list(filter(lambda x: name not in x.name, self._backbone.trainable_params()))
