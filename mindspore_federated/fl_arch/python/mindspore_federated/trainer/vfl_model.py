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
from mindspore_federated.privacy import LabelDP

from .vfl_optim import PartyOptimizer, PartyGradScaler, _reorganize_input_data


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
                 yaml_data=None,
                 grad_network=None):
        self._role = role
        self._backbone_net = network
        self._loss_fn = loss_fn
        self._grad_network = grad_network
        self._metrics = metrics
        self._eval_network = eval_network
        self._eval_indexes = eval_indexes
        self._label_dp = None

        self._yaml_data = yaml_data
        self._train_net_yaml = self._yaml_data['model']['train_net']
        self._eval_net_yaml = self._yaml_data['model']['eval_net']

        if 'privacy' in self._yaml_data:
            if 'label_dp' in self._yaml_data['privacy']:
                label_dp_eps = self._yaml_data['privacy']['label_dp']['eps']
                self._label_dp = LabelDP(eps=label_dp_eps)

        if self._label_dp is not None and self._role != 'leader':
            raise AttributeError('FLModel: only a leader party can employ the label dp strategy')
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

        if self._grad_network is None:
            if optimizers is None:
                self._optimizers = []
                self._build_optimizer()
            elif isinstance(optimizers, list):
                self._optimizers = optimizers
            else:
                self._optimizers = [optimizers]

            if 'grad_scalers' in self._yaml_data:
                self._grad_scalers = self._build_grad_scaler()
            else:
                self._grad_scalers = None

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
        for optim_data in self._yaml_data['opts']:
            optim_inst = PartyOptimizer(optim_data, self._train_network, self._train_net_yaml)
            self._optimizers.append(optim_inst)

    def _build_grad_scaler(self):
        """
        building the grad scale calulator of the leader/active party using the info. parsed from the yaml file.
        """
        if self._role == 'follower':
            raise AttributeError('follower party cannot call _build_grad_scaler')
        if 'grad_scalers' not in self._yaml_data:
            raise ValueError('info of grad_scaler is not defined in the yaml')
        grad_scalers = []
        for grad_scaler_yaml in self._yaml_data['grad_scalers']:
            grad_scalers.append(PartyGradScaler(grad_scaler_yaml, self._train_network, self._train_net_yaml))
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
        for input_data in self._eval_net_yaml['inputs']:
            if input_data['name'] in data_batch:
                input_data_batch[input_data['name']] = data_batch[input_data['name']]
            else:
                raise ValueError('missing input data \'%s\'' % input_data['name'])
        out_tuple = self._eval_network(**input_data_batch)
        if len(self._eval_net_yaml['outputs']) != len(out_tuple):
            raise ValueError('output of %s do not match the description of yaml' % self._eval_network.__name__)
        if self._eval_net_yaml['gt'] not in data_batch:
            raise ValueError('the label \'%s\'descripped in the yaml do not exist' % self._eval_net_yaml['gt'])
        if eval_metric is None:
            raise ValueError('not specify eval_metric')
        eval_metric.update(*out_tuple, data_batch[self._eval_net_yaml['gt']])
        out = OrderedDict()
        idx = 0
        for output_data in self._eval_net_yaml['outputs']:
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
        for input_data in self._train_net_yaml['inputs']:
            if input_data['name'] in data_batch:
                input_data_batch[input_data['name']] = data_batch[input_data['name']]
            else:
                raise ValueError("missing input data \'%s\'" % input_data['name'])
        input_data_batch = tuple(input_data_batch.values())
        out_tuple = self._train_network(*input_data_batch)
        if len(self._train_net_yaml['outputs']) != len(out_tuple):
            raise ValueError(f'output of {self._train_network.__name__} do not match the description of yaml')

        out = OrderedDict()
        for idx, output_data in enumerate(self._train_net_yaml['outputs']):
            out[output_data['name']] = out_tuple[idx]
        return out

    def backward_one_step(self, local_data_batch: dict = None, remote_data_batch: dict = None, sens: dict = None):
        """
        backward the network using a data batch.
        """
        if self._role == 'leader' and self._label_dp is not None:
            label = local_data_batch[self._yaml_data['model']['eval_net']['gt']]
            dp_label = self._label_dp(label)
            local_data_batch[self._yaml_data['model']['eval_net']['gt']] = dp_label

        if self._grad_network:
            return self._grad_network(local_data_batch, remote_data_batch)

        scales = dict()
        if self._grad_scalers:
            for grad_scaler in self._grad_scalers:
                scale = grad_scaler(local_data_batch, remote_data_batch, sens)
                scales[grad_scaler.grad_scale_name()] = scale

        if self._optimizers:
            for optimizer in self._optimizers:
                if isinstance(optimizer, PartyOptimizer):  # standard optimizer
                    optimizer(local_data_batch, remote_data_batch, sens)
                else:  # customized optimizer
                    input_names = [input_name['name'] for input_name in self._train_net_yaml['inputs']]
                    input_data_batch = _reorganize_input_data([local_data_batch, remote_data_batch],
                                                              input_names, type(optimizer).__name__)
                    input_data_batch = list(input_data_batch.values())
                    optimizer(*input_data_batch, sens=sens)

        if self._role == 'leader' and self._label_dp is not None:
            local_data_batch[self._yaml_data['model']['eval_net']['gt']] = label

        return scales
