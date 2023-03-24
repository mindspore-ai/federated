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
import os
import glob
from collections import OrderedDict

import mindspore
from mindspore import save_checkpoint
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Tensor, Parameter
from mindspore import nn, context
from mindspore.ops import PrimitiveWithInfer, prim_attr_register
from mindspore.context import ParallelMode
from mindspore.common.api import _pynative_executor
from ..startup.compress_config import CompressConfig, NO_COMPRESS_TYPE

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
    High-level API for training and inference of the vertical federated learning. The FLModel groups networks,
    optimizers, and other data structures into a high-level object. Then the FLModel builds the vertical federated
    learning process according to the yaml file provided by the developer, and provides interfaces controlling the
    training and inference processes.

    Args:
        yaml_data (class): Data class containing information on the vertical federated learning process, including
            optimizers, gradient calculators, etc. The information mentioned above is parsed from the yaml file
            provided by the developer.
        network (Cell): Training network, which outputs the loss. If loss_fn is not specified, the
            network will be used as the training network directly. If `loss_fn` is specified, the training network
            will be constructed on the basis of `network` and `loss_fn`.
        loss_fn (Cell): Loss function. If not specified, the input network will be used as the training network.
            Default: None.
        optimizers (Cell): Customized optimizer for training the train_network. If `optimizers` is None, FLModel will
            try to use standard optimizers of MindSpore specified in the yaml file. Default: None.
        metrics (Metric): Metrics to evaluate the evaluation network. Default: None.
        eval_network (Cell): Evaluation network of the party, which outputs the predict value. Default: None.

    Examples:
        >>> from mindspore_federated import FLModel, FLYamlData
        >>> import mindspore.nn as nn
        >>> yaml_data = FLYamlData(os.path.join(os.getcwd(), 'net.yaml'))
        >>> # define the training network
        >>> train_net = TrainNet()
        >>> # define the evaluation network
        >>> eval_net = EvalNet()
        >>> eval_metric = nn.Accuracy()
        >>> party_fl_model = FLModel(yaml_data, train_net, metrics=eval_metric, eval_network=eval_net)
    """

    def __init__(self,
                 yaml_data,
                 network,
                 loss_fn=None,
                 optimizers=None,
                 metrics=None,
                 eval_network=None):
        self._yaml_data = yaml_data
        self._role = self._yaml_data.role
        self._train_net_yaml = self._yaml_data.train_net
        self._eval_net_yaml = self._yaml_data.eval_net
        self._ckpoint_path = self._yaml_data.ckpt_path

        if not isinstance(network, nn.Cell):
            raise TypeError('FLModel: type of \'network\' is not nn.Cell')
        self._train_network = network
        self._loss_fn = loss_fn
        if self._loss_fn is not None:
            self._train_network = self._build_train_network()

        if eval_network and not isinstance(eval_network, nn.Cell):
            raise TypeError('FLModel: type of \'eval_network\' is not nn.Cell')
        self._eval_network = eval_network

        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            self._train_network = _VirtualDatasetCell(self._train_network)
            self._train_network.set_auto_parallel()
            if self._eval_network:
                self._eval_network = _VirtualDatasetCell(self._eval_network)
                self._eval_network.set_auto_parallel()
        if parallel_mode == ParallelMode.DATA_PARALLEL:
            self._train_network.set_broadcast_flag()
        self._train_network.set_train(mode=True)

        self._metrics = metrics

        self.global_step = 0

        if optimizers is None:
            self._optimizers = []
            self._build_optimizer()
        elif isinstance(optimizers, list):
            self._optimizers = optimizers
        else:
            self._optimizers = [optimizers]

        self._grad_scalers = self._build_grad_scaler() if self._yaml_data.grad_scalers else None

    def _build_train_network(self):
        """
        Build the network object using the input loss_fn and network.
        """
        network = nn.WithLossCell(self._train_network, self._loss_fn)
        net_inputs = self._train_network.get_inputs()
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
        Build the optimizer object using the information parsed from the yaml file.
        """
        for optim_data in self._yaml_data.opts:
            optim_inst = PartyOptimizer(optim_data, self._train_network, self._train_net_yaml)
            self._optimizers.append(optim_inst)

    def _build_grad_scaler(self):
        """
        Building the grad scale calculator of the party using the information parsed from the yaml file.
        """
        grad_scalers = []
        for grad_scaler_yaml in self._yaml_data.grad_scalers:
            grad_scalers.append(PartyGradScaler(grad_scaler_yaml, self._train_network, self._train_net_yaml))
        return grad_scalers

    def eval_one_step(self, local_data_batch: dict = None, remote_data_batch: dict = None):
        """
        Execute the evaluation network using a data batch.

        Args:
            local_data_batch (dict): Data batch read from local server. Key is the name of the data item, Value
                is the corresponding tensor.
            remote_data_batch (dict): Data batch read from remote server of other parties. Key is the name of
                the data item, Value is the corresponding tensor.

        Returns:
            Dict, outputs of the evaluation network. Key is the name of output, Value is tensors.

        Examples:
            >>> party_fl_model.eval_one_step(eval_item, embedding)
        """
        item_list = []
        if isinstance(local_data_batch, dict):
            item_list.extend(local_data_batch.items())
        if isinstance(remote_data_batch, dict):
            item_list.extend(remote_data_batch.items())
        data_batch = dict(item_list)
        input_data_batch = OrderedDict()
        for input_data in self._yaml_data.eval_net_ins:
            if input_data['name'] in data_batch:
                input_data_batch[input_data['name']] = data_batch[input_data['name']]
            else:
                raise ValueError('FLModel: missing input data \'%s\'' % input_data['name'])
        input_data_batch = tuple(input_data_batch.values())
        out_tuple = self._eval_network(*input_data_batch)
        if len(self._yaml_data.eval_net_outs) != len(out_tuple):
            raise ValueError('FLModel: output of %s do not match the description of yaml' % self._eval_network.__name__)
        if self._yaml_data.eval_net_gt not in data_batch:
            raise ValueError('FLModel: the label \'%s\'described in the yaml do not exist'
                             % self._yaml_data.eval_net_gt)
        if self._metrics is None:
            raise AttributeError('FLModel: try to execute eval_one_step but not specify eval_metric')
        self._metrics.update(*out_tuple, data_batch[self._yaml_data.eval_net_gt])
        out = OrderedDict()
        idx = 0
        for output_data in self._yaml_data.eval_net_outs:
            out[output_data['name']] = out_tuple[idx]
            idx += 1
        return out

    @staticmethod
    def _get_compress_config(items):
        """get compress config"""
        compress_configs = dict()
        for item in items:
            if 'name' in item:
                name = item['name']
            else:
                raise ValueError("Field 'name' is missing.")
            if 'compress_type' in item:
                compress_type = item['compress_type']
                if compress_type != NO_COMPRESS_TYPE:
                    bit_num = item.get('bit_num', 8)
                    compress_config = CompressConfig(compress_type, bit_num)
                    compress_configs[name] = compress_config
        return compress_configs

    def get_compress_configs(self):
        """
        Load the communication compression configs set in `yaml_data`, and return the configs for communicator.

        Note:
            Cannot use different compress methods if the names of tensors are the same.

        Returns:
            Dict, Key is the name of tensor, Value is the tensor.

        Examples:
            >>> compress_configs = party_fl_model.get_compress_configs()
        """
        train_in_compress_configs = self._get_compress_config(self._yaml_data.train_net_ins)
        train_out_compress_configs = self._get_compress_config(self._yaml_data.train_net_outs)
        eval_in_compress_configs = self._get_compress_config(self._yaml_data.eval_net_ins)
        eval_out_compress_configs = self._get_compress_config(self._yaml_data.eval_net_outs)

        compress_configs = {**train_in_compress_configs, **train_out_compress_configs,
                            **eval_in_compress_configs, **eval_out_compress_configs}
        return compress_configs

    def forward_one_step(self, local_data_batch: dict = None, remote_data_batch: dict = None):
        """
        Forward the training network using a data batch.

        Args:
            local_data_batch (dict): Data batch read from local server. Key is the name of the data item, Value
                is the corresponding tensor.
            remote_data_batch (dict): Data batch read from remote server of other parties. Key is the name of
                the data item, Value is the corresponding tensor.

        Returns:
            Dict, outputs of the training network. Key is the name of output, Value is the tensor.

        Examples:
            >>> logit_out = party_fl_model.forward_one_step(item, backbone_out)
        """
        item_list = []
        if isinstance(local_data_batch, dict):
            item_list.extend(local_data_batch.items())
        if isinstance(remote_data_batch, dict):
            item_list.extend(remote_data_batch.items())
        data_batch = dict(item_list)
        input_data_batch = OrderedDict()
        for input_data in self._yaml_data.train_net_ins:
            if input_data['name'] in data_batch:
                input_data_batch[input_data['name']] = data_batch[input_data['name']]
            else:
                raise ValueError("FLModel: missing input data \'%s\'" % input_data['name'])
        input_data_batch = tuple(input_data_batch.values())
        out_tuple = self._train_network(*input_data_batch)
        out_length = len(out_tuple) if isinstance(out_tuple, tuple) else 1
        if out_length != len(self._yaml_data.train_net_outs):
            raise ValueError('FLModel: output of %s do not match the description of yaml'
                             % self._train_network.__name__)

        out = OrderedDict()
        for idx, output_data in enumerate(self._yaml_data.train_net_outs):
            out[output_data['name']] = out_tuple[idx] if isinstance(out_tuple, tuple) else out_tuple

        return out

    def backward_one_step(self, local_data_batch: dict = None, remote_data_batch: dict = None, sens: dict = None):
        """
        Backward the training network using a data batch.

        Args:
            local_data_batch (dict): Data batch read from local server. Key is the name of the data item, Value
                is the corresponding tensor.
            remote_data_batch (dict): Data batch read from remote server of other parties. Key is the name of
                the data item, Value is the corresponding tensor.
            sens (dict): Sense parameters or scale values to calculate the gradient values of the training network.
                Key is the label name specified in the yaml file. Value is the dict of sense parameters
                or gradient scale values. the Key of the Value dict is the name of the output of
                the training network, and the Value of the Value dict is the sense tensor of corresponding output.

        Returns:
            Dict, sense parameters or gradient scale values sending to other parties. Key is the label name specified
            in the yaml file. Value is the dict of sense parameters or gradient scale values. the Key of the Value dict
            is the input of the training network, and the Value of the Value dict is the sense tensor of corresponding
            input.

        Examples:
            >>> head_scale = party_fl_model.backward_one_step(item, backbone_out)
        """
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
                    input_names = [input_name['name'] for input_name in self._yaml_data.train_net_ins]
                    input_data_batch = _reorganize_input_data([local_data_batch, remote_data_batch],
                                                              input_names, type(optimizer).__name__)
                    input_data_batch = list(input_data_batch.values())
                    if not sens:
                        optimizer(*input_data_batch)
                    else:
                        optimizer(*input_data_batch, sens=sens)

        # increase the global step
        self.global_step += 1
        # clean pynative cache to avoid memory leaks
        if self.global_step % 100 == 0:
            _pynative_executor.sync()

        return scales

    def save_ckpt(self, path: str = None):
        """
        Save checkpoints of the training network.

        Args:
            path (str): Path to save the checkpoint. If not specified, using the ckpt_path specified in the
                yaml file. Default: None.

        Examples:
            >>> party_fl_model.save_ckpt("party_fl_model.ckpt")
        """
        if path is not None:
            abs_path = os.path.abspath(path)
            self._ckpoint_path = abs_path

        if not os.path.exists(self._ckpoint_path):
            os.makedirs(self._ckpoint_path, exist_ok=True)

        prefix = self._yaml_data.train_net['name']
        cur_ckpoint_file = prefix + "_step_" + str(self.global_step) + ".ckpt"
        cur_file = os.path.join(self._ckpoint_path, cur_ckpoint_file)
        append_dict = {}
        append_dict['global_step'] = self.global_step
        save_checkpoint(self._train_network, cur_file, append_dict=append_dict)

    def load_ckpt(self, phrase: str = 'eval', path: str = None):
        """
        Load checkpoints for the training network and the evaluation network.

        Args:
            phrase (str): Load checkpoint to either training network (if set 'eval') or evaluation network (if set
                'train').  Default: 'eval'.
            path (str): Path to load the checkpoint. If not specified, using the ckpt_path specified in the
                yaml file. Default: None.

        Examples:
            >>> party_fl_model.load_ckpt(phrase="eval", path="party_fl_model.ckpt")
        """
        need_load_ckpt = None
        if path is None:
            if not os.path.exists(self._ckpoint_path):
                raise AttributeError(
                    'FLModel: not specify path and \'ckpt_path\' in the yaml do not exist, please check it!')
            last_file_path = self._get_last_file(self._ckpoint_path)
            need_load_ckpt = last_file_path
        else:
            file_path = os.path.abspath(path)
            if os.path.isfile(file_path):
                need_load_ckpt = file_path
            if os.path.isdir(file_path):
                last_file_path = self._get_last_file(file_path)
                need_load_ckpt = last_file_path
            if not os.path.exists(path):
                raise AttributeError('FLModel: {} do not exits, please check it!'.format(path))

        param_dict = load_checkpoint(need_load_ckpt)
        if phrase == 'train':
            _ = load_param_into_net(self._train_network, param_dict, strict_load=True)
            cur_global_step = param_dict['global_step'].asnumpy().tolist()
            self.global_step = cur_global_step
            for opt in self._optimizers:
                opt.optimizer.global_step = Parameter(Tensor((self.global_step,), mindspore.int32))
        elif phrase == 'eval':
            _ = load_param_into_net(self._eval_network, param_dict, strict_load=True)
        else:
            raise AttributeError('FLModel: phase must be \'train\' or \'eval\', please check it!')

    def _get_last_file(self, path: str):
        file_lists = os.listdir(path)
        if not file_lists:
            raise AttributeError('FLModel: {} is empty, please check it!'.format(path))
        file_lists = glob.glob(path + "/" + str(self._yaml_data.train_net['name']) + "*")
        if not file_lists:
            raise AttributeError('FLModel: in folder {}, no file starts with the name of training network ({}), \
                please check it!'.format(path, self._yaml_data.train_net['name']))
        file_need = max(file_lists, key=os.path.getctime)
        return file_need
