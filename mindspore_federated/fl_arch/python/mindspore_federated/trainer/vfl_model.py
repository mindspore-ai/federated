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
"""Classes and methods for modeling network of vfl parties in the scenario of splitnn."""

from collections import OrderedDict

from mindspore import nn, context
from mindspore.ops import PrimitiveWithInfer, prim_attr_register
from mindspore.context import ParallelMode
from mindspore.nn.metrics import Metric

from ..common import vfl_utils
from .vfl_optim import PartyGradOperation, PartyOptimizer, PartyGradScaler


class _VirtualDataset(PrimitiveWithInfer):
    """
    Auto parallel virtual dataset operator.

    It would insert VirtualDataset operator in forward computation and be deleted before backward computation.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize _VirtualDataset."""

    def infer_shape(self, *args):
        return args

    def infer_dtype(self, *args):
        return args


class _VirtualDatasetCell(nn.Cell):
    """
    Wrap the network with virtual dataset to convert data parallel layout to model parallel layout.

    _VirtualDataset is a virtual Primitive, it does not exist in the final executing graph. Inputs and outputs
    of _VirtualDataset are distributed in data parallel pattern, tensor redistribution Primitives is inserted
    dynamically during the graph compile process.

    Note:
        Only used in semi auto parallel and auto parallel mode.

    Args:
        backbone (Cell): The target network to wrap.

    Examples:
        >>> net = Net()
        >>> net = _VirtualDatasetCell(net)
    """

    def __init__(self, backbone):
        super(_VirtualDatasetCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._virtual_dataset = _VirtualDataset()

    def construct(self, *inputs):
        output = self._virtual_dataset(*inputs)
        return self._backbone(*output)


class FLModel:
    """
    Construct the training and evaluating process of split-learning using the yaml file and the nn.Cell provided
    by the user. The interface of the class follows the Model of mindspore, and try to encapulate the communication
    and weight updating processes.

    Args:
        role (enum): role of the party.
        network (nn.Cell): the backbone network of the party.
        train_network (nn.Cell): the loss network of the party.
        loss_fn  (nn.Cell): the loss_fn of the party, no need to provide if specified train_network
        optimizer (nn.Cell): the optimizer of the party, no need to provide if specified in the yaml file.
        metrics (class): metrics class of the party, used in the evaluation process.
        eval_network (nn.Cell): the evaluation network of the party, alternative to provide.
        eval_indexes (union[tuple, list, int]): indexes of data used in evaluation.
        yaml_data (dict): data structure describing info. of optimizers, grad calculator, communication and etc,
            parsing from the yaml file.
    """

    def __init__(self,
                 role,
                 network,
                 train_network,
                 loss_fn=None,
                 optimizers=None,
                 metrics=None,
                 eval_network=None,
                 eval_indexes=None,
                 yaml_data=None):
        self._role = role
        self._backbone_net = network
        self._loss_fn = loss_fn
        self._metrics = metrics
        self._eval_network = eval_network
        self._eval_indexes = eval_indexes

        self._yaml_data = yaml_data
        self._train_net_yaml_data = self._yaml_data['model']['train_net']
        self._eval_net_yaml_data = self._yaml_data['model']['eval_net']

        if train_network is not None and self._loss_fn is not None:
            raise AttributeError('FLModel: the attribute train_network and loss_fn shall be selected only one')
        if train_network is None and self._loss_fn is not None:
            self._train_network = self._build_train_network()
        elif train_network is not None and self._loss_fn is None:
            self._train_network = train_network
        else:
            self._train_network = network
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            self._train_network = _VirtualDatasetCell(self._train_network)
            self._eval_network = _VirtualDatasetCell(self._eval_network)
            self._train_network.set_auto_parallel()
            self._eval_network.set_auto_parallel()
        if parallel_mode == ParallelMode.DATA_PARALLEL:
            self._train_network.set_broadcast_flag()
        self._train_network.set_train(mode=True)

        if optimizers is None:
            self._optimizers = []
            self._build_optimizer()
        else:
            self._optimizers = optimizers

        if self._role == 'leader':
            self._grad_scalers = self._build_grad_scaler()

    def _build_train_network(self):
        """
        build the network object using the input loss_fn and network.
        """
        network = nn.WithLossCell(self._backbone_net, self._loss_fn)
        net_inputs = self._backbone_net.get_inputs()
        loss_inputs = [None]
        if self._loss_fn:
            if self._loss_fn.get_inputs():
                loss_inputs = [*self._loss_fn.get_inputs()]
            loss_inputs.pop(0)
            if net_inputs:
                net_inputs = [*net_inputs, *loss_inputs]
        if net_inputs is not None:
            network.set_inputs(*net_inputs)
        return network

    def _build_optimizer(self):
        """
        build the optimizer object using the info. parsed from the yaml file.
        """
        if self._yaml_data is None:
            raise AttributeError("yaml_data is required to build GrapOperation and Optimizer")
        for opt_data in self._yaml_data['opts']:
            grad_list = []
            weight_name_list = [param['name'] for param in opt_data['params']]
            params = vfl_utils.get_params_by_name(self._train_network, weight_name_list)
            for grad_data in opt_data['grads']:
                grad_op = PartyGradOperation(self._train_network, grad_data, params)
                grad_list.append(grad_op)
            opt_op = PartyOptimizer(opt_data['type'], self._train_network, params, opt_data['hyper_parameters'],
                                    grad_list)
            self._optimizers.append(opt_op)

    def _build_grad_scaler(self):
        """
        building the grad scale calulator of the leader/active party using the info. parsed from the yaml file.
        """
        if self._role == 'follower':
            raise AttributeError('follower party cannot call _build_grad_scaler')
        if 'grad_scales' not in self._yaml_data:
            raise ValueError('info of grad_scaler is not defined in the yaml')
        grad_scalers = []
        for grad_scaler_data in self._yaml_data['grad_scales']:
            grad_scalers.append(PartyGradScaler(self._train_network, grad_scaler_data))
        return grad_scalers

    def eval_one_step(self, local_data_batch: dict = None, remote_data_batch: dict = None, eval_metric: Metric = None):
        """
        evaluate the trained network using a data batch.
        """
        item_list = []
        if isinstance(local_data_batch, dict):
            item_list.extend(local_data_batch.items())
        if isinstance(remote_data_batch, dict):
            item_list.extend(remote_data_batch.items())
        data_batch = dict(item_list)
        input_data_batch = OrderedDict()
        for input_data in self._eval_net_yaml_data['inputs']:
            if input_data['name'] in data_batch:
                input_data_batch[input_data['name']] = data_batch[input_data['name']]
            else:
                raise ValueError('missing input data \'%s\'' % input_data['name'])
        out_tuple = self._eval_network(**input_data_batch)
        if len(self._eval_net_yaml_data['outputs']) != len(out_tuple):
            raise ValueError('output of %s do not match the description of yaml' % self._eval_network.__name__)
        if self._eval_net_yaml_data['gt'] not in data_batch:
            raise ValueError('the label \'%s\'descripped in the yaml do not exist' % self._eval_net_yaml_data['gt'])
        if eval_metric is None:
            raise ValueError('not specify eval_metric')
        eval_metric.update(*out_tuple, data_batch[self._eval_net_yaml_data['gt']])
        out = OrderedDict()
        idx = 0
        for output_data in self._eval_net_yaml_data['outputs']:
            out[output_data['name']] = out_tuple[idx]
            idx += 1
        return out

    def forward_one_step(self, local_data_batch: dict = None, remote_data_batch: dict = None):
        """
        forward the network using a data batch.
        """
        item_list = []
        if isinstance(local_data_batch, dict):
            item_list.extend(local_data_batch.items())
        if isinstance(remote_data_batch, dict):
            item_list.extend(remote_data_batch.items())
        data_batch = dict(item_list)
        input_data_batch = OrderedDict()
        for input_data in self._train_net_yaml_data['inputs']:
            if input_data['name'] in data_batch:
                input_data_batch[input_data['name']] = data_batch[input_data['name']]
            else:
                raise ValueError("missing input data \'%s\'" % input_data['name'])
        input_data_batch = tuple(input_data_batch.values())
        out_tuple = self._train_network(*input_data_batch)
        if len(self._train_net_yaml_data['outputs']) != len(out_tuple):
            raise ValueError(f'output of {self._train_network.__name__} do not match the description of yaml')

        out = OrderedDict()
        for idx, output_data in enumerate(self._train_net_yaml_data['outputs']):
            out[output_data['name']] = out_tuple[idx]
        return out

    def backward_one_step(self, local_data_batch: dict = None, remote_data_batch: dict = None, sens: dict = None):
        """
        backward the network using a data batch.
        """
        scales = dict()
        if self._role == 'leader' and self._grad_scalers:
            for grad_scaler in self._grad_scalers:
                scale = grad_scaler(local_data_batch, remote_data_batch)
                scales[grad_scaler.grad_scale_name()] = scale
        if self._optimizers:
            for optimizer in self._optimizers:
                optimizer(local_data_batch, remote_data_batch, sens)
        return scales
