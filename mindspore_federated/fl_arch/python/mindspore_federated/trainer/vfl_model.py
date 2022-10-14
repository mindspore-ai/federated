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
    High-level API for training and inference of the vertical federated learning. The FLModel groups networks,
    optimizers, and other data structures into a high-level object. Then the FLModel builds the vertical federated
    learning process according to the yaml file provided by the developer, and provides interfaces controlling the
    training and inference processes.

    Args:
        role (str): Role of the vertical federated learning party, shall be either 'leader' or 'follower'.
        yaml_data (class): Data class containing information on the vertical federated learning process, including
            optimizers, gradient calculators, etc. The information mentioned above is parsed from the yaml file
            provided by the developer.
        network (Cell): Backbone network of the party, weights of which will be shared by the training network
            and the evaluation network.
        train_network (Cell): Training network of the party, which outputs the loss. If not specified, FLModel
            will construct the training network using the input network and loss_fn. Default: None.
        loss_fn (Cell): Loss function to construct the training network on the basis of the input network. If a
            train_network has been specified, it will not work even has been provided. Default: None.
        optimizer (Cell): Customized optimizer for training the train_network. If not specified, FLModel will try
            to use standard optimizers of MindSpore specified in the yaml file. Default: None.
        metrics (Metric): Metrics to evaluate the evaluation network. Default: None.
        eval_network (nn.Cell): Evaluation network of the party, which outputs the predict value. Default: None.
    """

    def __init__(self,
                 role,
                 yaml_data,
                 network,
                 train_network=None,
                 loss_fn=None,
                 optimizers=None,
                 metrics=None,
                 eval_network=None,
                 grad_network=None):
        self._role = role
        self._backbone_net = network
        self._loss_fn = loss_fn
        self._grad_network = grad_network
        self._metrics = metrics
        self._eval_network = eval_network
        self._label_dp = None

        self._yaml_data = yaml_data
        self._train_net_yaml = self._yaml_data.train_net
        self._eval_net_yaml = self._yaml_data.eval_net

        if hasattr(self._yaml_data, 'privacy'):
            label_dp_eps = self._yaml_data.privacy_eps
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

            self._grad_scalers = self._build_grad_scaler() if self._yaml_data.grad_scalers else None

    def _build_train_network(self):
        """
        Build the network object using the input loss_fn and network.
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
        Build the optimizer object using the information parsed from the yaml file.
        """
        if self._yaml_data is None:
            raise AttributeError("yaml_data is required to build GrapOperation and Optimizer")
        for optim_data in self._yaml_data.opts:
            optim_inst = PartyOptimizer(optim_data, self._train_network, self._train_net_yaml)
            self._optimizers.append(optim_inst)

    def _build_grad_scaler(self):
        """
        Building the grad scale calulator of the party using the information parsed from the yaml file.
        """
        grad_scalers = []
        for grad_scaler_yaml in self._yaml_data.grad_scalers:
            grad_scalers.append(PartyGradScaler(grad_scaler_yaml, self._train_network, self._train_net_yaml))
        return grad_scalers

    def eval_one_step(self, local_data_batch: dict = None, remote_data_batch: dict = None):
        """
        Evaluate the evaluation network using a data batch.

        Args:
            local_data_batch (dict): Data batch read from local server.
            remote_data_batch (dict): Data batch read from remote server of other parties.

        Returns:
            Dict, outputs of the evaluation network. key is the name of output, value is tensors.
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
                raise ValueError('missing input data \'%s\'' % input_data['name'])
        out_tuple = self._eval_network(**input_data_batch)
        if len(self._yaml_data.eval_net_outs) != len(out_tuple):
            raise ValueError('output of %s do not match the description of yaml' % self._eval_network.__name__)
        if self._yaml_data.eval_net_gt not in data_batch:
            raise ValueError('the label \'%s\'descripped in the yaml do not exist' % self._yaml_data.eval_net_gt)
        if self._metrics is None:
            raise ValueError('not specify eval_metric')
        self._metrics.update(*out_tuple, data_batch[self._yaml_data.eval_net_gt])
        out = OrderedDict()
        idx = 0
        for output_data in self._yaml_data.eval_net_outs:
            out[output_data['name']] = out_tuple[idx]
            idx += 1
        return out

    def forward_one_step(self, local_data_batch: dict = None, remote_data_batch: dict = None):
        """
        Forward the training network using a data batch.

        Args:
            local_data_batch (dict): Data batch read from local server.
            remote_data_batch (dict): Data batch read from remote server of other parties.

        Returns:
            Dict, outputs of the training network. key is the name of output, value is the tensor.
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
                raise ValueError("missing input data \'%s\'" % input_data['name'])
        input_data_batch = tuple(input_data_batch.values())
        out_tuple = self._train_network(*input_data_batch)
        if len(self._yaml_data.train_net_outs) != len(out_tuple):
            raise ValueError(f'output of {self._train_network.__name__} do not match the description of yaml')

        out = OrderedDict()
        for idx, output_data in enumerate(self._yaml_data.train_net_outs):
            out[output_data['name']] = out_tuple[idx]
        return out

    def backward_one_step(self, local_data_batch: dict = None, remote_data_batch: dict = None, sens: dict = None):
        """
        Backward the training network using a data batch.

        Args:
            local_data_batch (dict): Data batch read from local server.
            remote_data_batch (dict): Data batch read from remote server of other parties.
            sens (dict): Sense parameters or scale values to calculate the gradient values of the traning network.
            key is the label name specified in the yaml file. value is the dict of sense parameters or gradient scale
            values. the key of the value dict is the name of the output of the training network, and the value of the
            value dict is the sense tensor of corresponding output.

        Returns:
            Dict, sense parameters or gradient scale values sending to other parties. key is the label name specified
            in the yaml file. value is the dict of sense parameters or gradient scale values. the key of the value dict
            is the input of the training network, and the value of the value dict is the sense tensor of corresponding
            input.
        """
        if self._role == 'leader' and self._label_dp is not None:
            label = local_data_batch[self._yaml_data.eval_net_gt]
            dp_label = self._label_dp(label)
            local_data_batch[self._yaml_data.eval_net_gt] = dp_label

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
                    input_names = [input_name['name'] for input_name in self._yaml_data.train_net_ins]
                    input_data_batch = _reorganize_input_data([local_data_batch, remote_data_batch],
                                                              input_names, type(optimizer).__name__)
                    input_data_batch = list(input_data_batch.values())
                    optimizer(*input_data_batch, sens=sens)

        if self._role == 'leader' and self._label_dp is not None:
            local_data_batch[self._yaml_data.eval_net_gt] = label

        return scales

    def save_ckpt(self, path: str = None):
        """
        Save checkpoints of the training network.

        Args:
            path (str): Path to save the checkpoint. If not specified, using the ckpt_path specified in the
                yaml file. Default: None.
        """
        print('save_ckpt to %s', path)

    def load_ckpt(self, phrase: str = 'eval', path: str = None):
        """
        Load checkpoints for the training network and the evaluation network.

        Args:
            phrase (str): Load checkpoint to either training network (if set 'eval') or evaluation network (if set
                'train').  Default: 'eval'.
            path (str): Path to load the checkpoint. If not specified, using the ckpt_path specified in the
                yaml file. Default: None.
        """
        print('load_ckpt to %s, %s', (phrase, path))
