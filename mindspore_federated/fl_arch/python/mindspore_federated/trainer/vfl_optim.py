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

from mindspore import context, nn, ops
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
        optim_yaml (tuple): data describing on the optimizer, parsed from the yaml file.
        net (nn.Cell): the network to be optimized, provided by the mindspore framework.
        net_yaml (tuple): data describing on the network, parsed from the yaml file.

    Inputs:
        local_data_batch (dict): data batch from local data sources.
        remote_data_batch (dict): data batch from remote data sources.
        sens_dict (dict): sens params for optimizer.

    Examples:
        >>> net = Net()
        >>> optim = PartyOptimizer(optim_yaml_data, net, net_yaml_data)
    """
    def __init__(self, optim_yaml: dict, net: nn.Cell, net_yaml: dict):
        self.type = optim_yaml['type']
        self.hyperparams = optim_yaml['hyper_parameters']
        self.net = net
        if 'params' in optim_yaml:  # if only optimize specified params
            param_name_list = [param['name'] for param in optim_yaml['params']]
            optim_params = vfl_utils.get_params_by_name(net, param_name_list)
        else:
            optim_params = net.trainable_params()
        self.hyperparams['params'] = self._params = optim_params
        optim_module = importlib.import_module('mindspore.nn.optim')
        self.optimizer = getattr(optim_module, self.type)(**self.hyperparams)
        self.grad_list = []
        for grad_yaml in optim_yaml['grads']:
            grad_inst = PartyGradOperation(grad_yaml, net, net_yaml, optim_params)
            self.grad_list.append(grad_inst)

    def __call__(self, local_data_batch: dict = None, remote_data_batch: dict = None, sens_dict: dict = None):
        grad_value = tuple()
        for grad in self.grad_list:
            if not grad_value:
                grad_value = grad(local_data_batch, remote_data_batch, sens_dict)
            else:
                res = grad(local_data_batch, remote_data_batch, sens_dict)
                zipped = zip(res, grad_value)
                grad_value = tuple(map(sum, zipped))
        self.optimizer(grad_value)


class PartyGradOperation:
    """
    GradOperation for network of vfl parties.

    Args:
        grad_yaml (dict): data describing on the grad calculating, parsed from the yaml file.
        net (nn.Cell): net to calculate grads, which input features and output loss values.
        net_yaml (dict): data describing on the training network, parsed from the yaml file.
        optim_params (dict): parameters of the network need to be optimized. If no params specified inside
                             the grad, PartyGradOperation will try to calculate grads of the input params.

    Examples:
        >>> net = Net()
        >>> params = net.trainable_params()
        >>> params = ParameterTuple(params)
        >>> grad = PartyGradOperation(grad_yaml_data, net, net_yaml_data, params)
    """
    def __init__(self, grad_yaml: dict, net: nn.Cell, net_yaml: dict, optim_params: tuple):
        self._name = 'PartyGradOperation_'.join(net_yaml['name'])
        self._net = net
        self._input_names = [input_name['name'] for input_name in grad_yaml['inputs']]
        self._output_name = grad_yaml['output']['name']
        if 'params' in grad_yaml:
            param_name_list = [param['name'] for param in grad_yaml['params']]
            self._params = vfl_utils.get_params_by_name(net, param_name_list)
        else:
            self._params = optim_params
        self._sens = grad_yaml['sens'] if 'sens' in grad_yaml else None
        self._sens_param = 'sens' in grad_yaml
        if len(net_yaml['outputs']) > 1:
            for idx, output in enumerate(net_yaml['outputs']):
                if output['name'] == self._output_name:
                    self._output_index = idx
                    break
            self._loss = vfl_utils.IthOutputCellInTuple(self._net, self._output_index)
        else:
            self._loss = self._net
        self._loss.set_grad()
        self._grad_op = ops.GradOperation(get_all=False, get_by_list=True, sens_param=self._sens_param)
        self._grad_op = self._grad_op(self._loss, self._params)
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        self.reducer_flag = parallel_mode == ParallelMode.DATA_PARALLEL
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = context.get_auto_parallel_context("device_num")
            self.grad_reducer = DistributedGradReducer(self._params, mean, degree)
        self._loss_dtype = None
        self._loss_shape = None
        self._dtype_op = ops.DType()
        self._shape_op = ops.Shape()
        self._fill_op = ops.Fill()

    def __call__(self, local_data_batch: dict = None, remote_data_batch: dict = None, sens: dict = None):
        input_data_batch = _reorganize_input_data([local_data_batch, remote_data_batch], self._input_names, self._name)
        input_data_batch = list(input_data_batch.values())
        if self._sens_param:
            if isinstance(self._sens, (float, int)):
                if not self._loss_dtype and not self._loss_shape:
                    loss_value = self._loss(*input_data_batch)
                    self._loss_dtype = self._dtype_op(loss_value)
                    self._loss_shape = self._shape_op(loss_value)
                sens_value = self._fill_op(self._loss_dtype, self._loss_shape, self._sens)
            elif sens is not None and isinstance(self._sens, str):
                sens_value = sens[self._sens]
                sens_value = sens_value[self._output_name]
            else:
                raise ValueError('Input a meaningless sens to %s' % self._name)
            grad_value = self._grad_op(*input_data_batch, sens_value)
        else:
            grad_value = self._grad_op(*input_data_batch)
        if self.reducer_flag:
            grad_value = self.grad_reducer(grad_value)
        return grad_value


class PartyGradScaler:
    """
    Calculate grad scales pass from the leader/active party to the follower/passive party.
    the grad scale value will be input as the sens of PartyGradOperation.

    Args:
        grad_scale_yaml (dict): data describing on the grad scale calculating, parsed from the yaml file.
        net (nn.Cell): loss net of the party, which input features and output loss values.
        net_yaml (dict): data describing on the training network, parsed from the yaml file.

    Examples:
        >>> net = Net()
        >>> grad_scaler = PartyGradScaler(scaler_yaml_data, net, net_yaml_data)
    """
    def __init__(self, grad_scale_yaml: dict, net: nn.Cell, net_yaml: dict):
        self._name = 'PartyGradScaler_'.join(net_yaml['name'])
        self._input_names = [input_name['name'] for input_name in grad_scale_yaml['inputs']]
        self._output_name = grad_scale_yaml['output']['name']
        if len(net_yaml['outputs']) > 1:
            for idx, output in enumerate(net_yaml['outputs']):
                if output['name'] == self._output_name:
                    self._output_index = idx
                    break
            self._loss_net = vfl_utils.IthOutputCellInTuple(net, self._output_index)
        else:
            self._loss_net = net
        self._net_yaml = net_yaml
        self._net_input_names = [input_name['name'] for input_name in net_yaml['inputs']]
        self._sens = grad_scale_yaml['sens'] if 'sens' in grad_scale_yaml else None  # indicator to sens
        self._sens_param = 'sens' in grad_scale_yaml  # whether use sens
        self._grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=self._sens_param)
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            self._loss_net = _VirtualDatasetCell(self._loss_net)
            self._loss_net.set_auto_parallel()
        self._loss_net.set_grad()
        self._sens_value = None
        self._dtype_op = ops.DType()
        self._shape_op = ops.Shape()
        self._fill_op = ops.Fill()

    def __call__(self, local_data_batch: dict = None, remote_data_batch: dict = None, sens: dict = None):
        input_data_batch = _reorganize_input_data([local_data_batch, remote_data_batch],
                                                  self._net_input_names, self._name)
        input_data_batch = tuple(input_data_batch.values())
        if not self._sens_param:
            grad_scale_value = self._grad_op(self._loss_net)(*input_data_batch)
        elif isinstance(self._sens, (float, int)):
            if not self._sens_value:
                loss_value = self._loss_net(*input_data_batch)
                self._fill_sens(loss_value)
            grad_scale_value = self._grad_op(self._loss_net)(*input_data_batch, self._sens_value)
        elif isinstance(self._sens, str):
            if self._sens not in sens:
                raise ValueError('Input sens of %s not containing %s' % (self._name, self._sens))
            sens_value = sens[self._sens][self._output_name]
            grad_scale_value = self._grad_op(self._loss_net)(*input_data_batch, sens_value)
        grad_scale_dict = OrderedDict()
        for input_name in self._input_names:
            if input_name in self._net_input_names:
                idx = self._net_input_names.index(input_name)
                grad_scale_dict[input_name] = grad_scale_value[idx]
        return grad_scale_dict

    def _fill_sens(self, loss):
        """
        generate sens matrix according to the shape of network output
        """
        if isinstance(loss, tuple):
            sens_list = []
            for loss_item in loss:
                loss_dtype = self._dtype_op(loss_item)
                loss_shape = self._shape_op(loss_item)
                sens_item = self._fill_op(loss_dtype, loss_shape, self._sens)
                sens_list.append(sens_item)
            self._sens_value = tuple(sens_list)
        else:
            loss_dtype = self._dtype_op(loss)
            loss_shape = self._shape_op(loss)
            self._sens_value = self._fill_op(loss_dtype, loss_shape, self._sens)

    def grad_scale_name(self):
        return self._output_name
