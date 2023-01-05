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
"""Classes and methods for modeling network of cross-silo hfl for multiple aggregation algorithm."""

from mindspore import nn
from mindspore.context import ParallelMode
from mindspore.train.model import Model
from mindspore.parallel._utils import _get_parallel_mode, _get_device_num, _get_global_rank, \
    _get_parameter_broadcast
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore_federated._mindspore_federated import FLContext
from ..aggregation.fedprox import WithFedProxLossCell
from ..common import _fl_context


class HFLModel(Model):
    """
    High-level API for training and inference of the horizontal federated learning. The HFLModel groups networks,
    optimizers, and other data structures into a high-level object. Then the HFLModel builds the horizontal federated
    learning process according to the yaml file provided by the developer, and provides interfaces controlling the
    training and inference processes.

    Args:
        network (Cell): Training network of the party, which outputs the loss. If loss_fn is not specified, the
            the network will be used as the training network directly. If loss_fn is specified, the training
            network will be constructed on the basis of network and loss_fn.
        loss_fn (Cell): Loss function to construct the training network on the basis of the input network. If a
            train_network has been specified, it will not work even has been provided. Default: None.
        optimizer (Cell): Customized optimizer for training the train_network. If not specified, FLModel will try
            to use standard optimizers of MindSpore specified in the yaml file. Default: None.
        metrics (Metric): Metrics to evaluate the evaluation network. Default: None.
        eval_network (nn.Cell): Evaluation network of the party, which outputs the predict value. Default: None.


    Examples:
        >>> from mindspore_federated.trainer.hfl_model import HFLModel
        >>> import mindspore.nn as nn
        >>> # define the network
        >>> network = MyNet()
        >>> net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> net_opt = nn.Momentum(network.trainable_params(), client_learning_rate, 0.9)
        >>> hfl_model = HFLModel(network, net_loss, net_opt)
    """

    def __init__(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None,
                 amp_level="O0", boost_level="O0", **kwargs):

        ctx = FLContext.get_instance()
        self._server_mode = ctx.server_mode()
        aggregation_type = ctx.aggregation_type()
        self._aggregation_type = aggregation_type
        if self._aggregation_type not in _fl_context.SUPPORT_AGG_TYPES and \
                self._server_mode == _fl_context.SERVER_MODE_CLOUD:
            raise ValueError(
                "aggregation_type must be in {}, but got {}.".format(_fl_context.SUPPORT_AGG_TYPES,
                                                                     self._aggregation_type))
        if self._aggregation_type == _fl_context.FEDPROX:
            self._iid_rate = ctx.iid_rate()

        self._network = network
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._loss_scale_manager = None
        self._loss_scale_manager_set = False
        self._keep_bn_fp32 = None
        self._check_kwargs(kwargs)
        self._amp_level = amp_level
        self._boost_level = boost_level
        self._eval_network = eval_network
        self._process_amp_args(kwargs)
        self._parallel_mode = _get_parallel_mode()
        self._device_number = _get_device_num()
        self._global_rank = _get_global_rank()
        self._parameter_broadcast = _get_parameter_broadcast()
        self._metrics = metrics

        self._check_amp_level_arg(optimizer, amp_level)
        self._check_for_graph_cell(kwargs)
        self._build_boost_network(kwargs)
        self._train_network = self._build_train_network()
        self._build_eval_network(metrics, self._eval_network, eval_indexes)
        self._build_predict_network()
        self._current_epoch_num = 0
        self._current_step_num = 0
        self.epoch_iter = 0
        self.enable_recovery = False
        self._backbone_is_train = True
        self.need_load_ckpt = False

    def _build_train_network(self):

        """Build train network"""
        network = self._network
        net_inputs = network.get_inputs()
        loss_inputs = [None]
        if self._loss_fn:
            if self._loss_fn.get_inputs():
                loss_inputs = [*self._loss_fn.get_inputs()]
            loss_inputs.pop(0)
            if net_inputs:
                net_inputs = [*net_inputs, *loss_inputs]
            if self._aggregation_type == _fl_context.FEDPROX:
                network = WithFedProxLossCell(network, self._loss_fn, self._iid_rate)
            else:
                network = nn.WithLossCell(network, self._loss_fn)
        if self._optimizer:
            loss_scale = 1.0
            network = nn.TrainOneStepCell(network, self._optimizer, loss_scale).set_train()

        # If need to check if loss_fn is not None, but optimizer is None

        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            network.set_auto_parallel()
            if self._optimizer is None:
                # In this case, multiple optimizer(s) is supposed to be included in 'self._network'
                _set_multi_subgraphs()
        if net_inputs is not None:
            network.set_inputs(*net_inputs)
        return network
