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
"""classes and functions for optimizing weights of split-learning networks."""

import importlib
from collections import OrderedDict

from mindspore import context, nn, ops, ParameterTuple
from mindspore.ops import PrimitiveWithInfer, prim_attr_register
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

from ..common import vfl_utils


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

def _reorganize_input_data(data_batch_list, data_name_list, module_name):
    """
    adjust the sequence of input data according to the yaml file.
    """
    item_list = []
    for data_batch in data_batch_list:
        if isinstance(data_batch, dict):
            item_list.extend(data_batch.items())
    data_batch = dict(item_list)
    input_data_batch = OrderedDict()
    for data_name in data_name_list:
        if data_name in data_batch:
            input_data_batch[data_name] = data_batch[data_name]
        else:
            raise ValueError('missing input data \'%s\' in PartyGradScaler of %s' % (data_name, module_name))
    return input_data_batch


class PartyOptimizer:
    """
    Optimizer for network of vfl parties.

    Args:
        name (str): type of the optimizer.
        net (nn.Cell): the network to be optimized, provided by the mindspore framework.
        params (tuple): parameters of the network to be optimized.
        hyperparams (dict): hyperparams of the optimizer.
        grad_list: the grad list calculating using PartyGradOperation.
    """
    def __init__(self, name: str, net: nn.Cell, params: tuple, hyperparams: dict, grad_list: list):
        optim_module = importlib.import_module('mindspore.nn.optim')
        self.hyperparams = hyperparams
        self.net = net
        self.hyperparams['params'] = self._params = params
        self.optim = getattr(optim_module, name)(**self.hyperparams)
        self.type = name
        self.grad_list = grad_list

    def __call__(self, local_data_batch: dict = None, remote_data_batch: dict = None, sens_dict: dict = None):
        grad_value = tuple()
        for grad in self.grad_list:
            if not grad_value:
                grad_value = grad(local_data_batch, remote_data_batch, sens_dict)
            else:
                res = grad(local_data_batch, remote_data_batch, sens_dict)
                zipped = zip(res, grad_value)
                grad_value = tuple(map(sum, zipped))
        self.optim(grad_value)


class PartyGradOperation:
    """
    GradOperation for network of vfl parties.

    Args:
        loss_net (nn.Cell): loss net of the party, which input features and output loss values.
        gra_data (dict): data describing on the grad calculating, parsed from the yaml file.
        params (dict): parameters of the network need to calculate grad values.
    """
    def __init__(self, loss_net: nn.Cell, grad_data: dict, params: tuple):
        self._name = 'PartyGradOperation'
        self._loss_net = loss_net
        self._input_names = [input_name['name'] for input_name in grad_data['inputs']]
        self._output_name = grad_data['output']['name']
        self._output_index = grad_data['output']['index']
        self._params = params
        self._sens = grad_data['sens'] if 'sens' in grad_data else None
        self._get_all = grad_data['get_all']
        self._get_by_list = grad_data['get_by_list']
        self._sens_param = grad_data['sens_param']
        self._loss = vfl_utils.IthOutputCellInTuple(self._loss_net, self._output_index)
        self._loss.set_grad()
        self._grad_op = ops.GradOperation(self._get_all, self._get_by_list, self._sens_param)
        self._grad_op = self._grad_op(self._loss) if (self._get_all and not self._get_by_list) \
            else self._grad_op(self._loss, self._params)
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        self.reducer_flag = parallel_mode == ParallelMode.DATA_PARALLEL
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = context.get_auto_parallel_context("device_num")
            self.grad_reducer = DistributedGradReducer(self._params, mean, degree)

    def __call__(self, local_data_batch: dict = None, remote_data_batch: dict = None, sens: dict = None):
        input_data_batch = _reorganize_input_data([local_data_batch, remote_data_batch], self._input_names, self._name)
        if self._sens_param:
            input_data_batch = list(input_data_batch.values())
            if isinstance(self._sens, float):
                loss_value = self._loss(*input_data_batch)
                sens_value = ops.Fill()(ops.DType()(loss_value), ops.Shape()(loss_value), self._sens)
            elif sens is not None and isinstance(self._sens, str):
                sens_value = sens[self._sens]
                sens_value = sens_value[self._output_name]
            else:
                raise ValueError('Not input meaningful sens value')
            grad_value = self._grad_op(*input_data_batch, sens_value)
            if self.reducer_flag:
                grad_value = self.grad_reducer(grad_value)
            return grad_value
        grad_value = self._grad_op(input_data_batch)
        if self.reducer_flag:
            grad_value = self.grad_reducer(grad_value)
        return grad_value


class PartyGradScaler:
    """
    Calculate grad scales pass from the leader/active party to the follower/passive party.
    the grad scale value will be input as the sens of PartyGradOperation.

    Args:
        loss_net (nn.Cell): loss net of the party, which input features and output loss values.
        grad_scale_data (dict): data describing on the grad scale calculating, parsed from the yaml file
        get_all (bool): whether calculating grad scale of all parameters.
        get_by_list (bool): whether calculating grad scale of specific parameters.
        sens_param (bool): whether specifying sens of grad.
    """
    def __init__(self, loss_net: nn.Cell, grad_scale_data: dict, get_all: bool = True, get_by_list: bool = False,
                 sens_param: bool = False):
        self._name = grad_scale_data['return']
        self._loss_net = loss_net
        self._input_names = [input_name['name'] for input_name in grad_scale_data['inputs']]
        self._output_name = grad_scale_data['output']['name']
        self._output_idx = grad_scale_data['output']['output_index']
        self._sens = grad_scale_data['sens']
        self._get_all = get_all
        self._get_by_list = get_by_list
        self._params = ParameterTuple(loss_net.trainable_params())
        self._sens_param = True if isinstance(self._sens, float) else sens_param
        self._grad_op = ops.GradOperation(self._get_all, self._get_by_list, self._sens_param)
        self._loss = vfl_utils.IthOutputCellInTuple(self._loss_net, self._output_idx)
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            self._loss = _VirtualDatasetCell(self._loss)
            self._loss.set_auto_parallel()
        self._loss.set_grad()

    def __call__(self, local_data_batch: dict = None, remote_data_batch: dict = None):
        input_data_batch = _reorganize_input_data([local_data_batch, remote_data_batch], self._input_names, self._name)
        if self._sens_param:
            input_data_batch = tuple(input_data_batch.values())
            loss_value = self._loss(*input_data_batch)
            sens_value = ops.Fill()(ops.DType()(loss_value), ops.Shape()(loss_value), self._sens)
            grad_scale_value = self._grad_op(self._loss)(*input_data_batch, sens_value)
        else:
            grad_scale_value = self._grad_op(self._loss)(*input_data_batch)
        remote_data_names = remote_data_batch.keys()
        remote_grad_scale_keys = []
        remote_grad_scale_values = []
        for remote_data_name in remote_data_names:
            if remote_data_name in self._input_names:
                idx = self._input_names.index(remote_data_name)
                remote_grad_scale_keys.append(remote_data_name)
                remote_grad_scale_values.append(grad_scale_value[idx])
        return OrderedDict(zip(remote_grad_scale_keys, remote_grad_scale_values))

    def grad_scale_name(self):
        return self._name
