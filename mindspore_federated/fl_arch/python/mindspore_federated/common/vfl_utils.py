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
"""Essential tools to modeling the split-learning process."""

import os.path

import yaml
from mindspore import nn, ParameterTuple


def parse_yaml_file(file_path):
    yaml_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        assert ValueError(f'File {yaml_path} not exit')
    with open(yaml_path, 'r', encoding='utf-8') as fp:
        yaml_data = yaml.safe_load(fp)
    return yaml_data, fp


class FLYamlData:
    """
    Data class storing configuration information on the vertical federated learning process, including
    inputs, outputs, and hyper-parameters of networks, optimizers, operators, etc. The information
    mentioned above is parsed from the yaml file provided by the developer of the vertical federated
    learning system. The class will verify the yaml file in the parsing process. The return value is
    used for the first input of FLModel.

    Args:
        yaml_path (str): Path of the yaml file.

    Examples:
        >>> from mindspore_federated import FLYamlData
        >>> yaml_data = FLYamlData(os.path.join(os.getcwd(), 'net.yaml'))
    """

    def __init__(self, yaml_path: str):
        yaml_path = os.path.abspath(yaml_path)
        if not os.path.exists(yaml_path):
            raise ValueError(f'File {yaml_path} not exit')
        with open(yaml_path, 'r', encoding='utf-8') as self.fp:
            self.yaml_data = yaml.safe_load(self.fp)
        try:
            if 'role' not in self.yaml_data:
                raise ValueError('FLYamlData init failed: missing field of \'role\'')
            self.role = self.yaml_data['role']
            if self.role not in ['leader', 'follower']:
                raise ValueError(f'FLYamlData init failed: value of role ({self.role}) is illegal, \
                                role shall be either \'leader\' or \'follower\'')

            if 'model' not in self.yaml_data:
                raise ValueError('FLYamlData init failed: missing field of \'model\'')
            self.model_data = self.yaml_data['model']
            self._parse_train_net()
            self._parse_eval_net()

            if 'opts' not in self.yaml_data:
                raise ValueError('FLYamlData init failed: missing field of \'opts\'')
            self.opts = self.yaml_data['opts']
            self._check_opts()

            self.grad_scalers = self.yaml_data['grad_scalers'] if 'grad_scalers' in self.yaml_data else None
            if self.grad_scalers:
                self._check_grad_scalers()

            self._parse_dataset()
            self._parse_hyper_params()

            self._parse_privacy()

            if 'ckpt_path' in self.yaml_data:
                self.ckpt_path = self.yaml_data['ckpt_path']
            else:
                self.ckpt_path = './checkpoints'
        finally:
            self.fp.close()

    def _parse_dataset(self):
        """Parse information on dataset."""
        if 'dataset' in self.yaml_data:
            dataset = self.yaml_data['dataset']
            self.dataset_name = dataset['name'] if 'name' in dataset else ""
            self.features = dataset['features'] if 'features' in dataset else []
            self.labels = dataset['labels'] if 'labels' in dataset else []

    def _parse_hyper_params(self):
        """Parse information on training hyper-parameters."""
        if 'hyper_parameters' in self.yaml_data:
            train_hyper_parameters = self.yaml_data['hyper_parameters']
            self.epochs = train_hyper_parameters['epochs'] if 'epochs' in train_hyper_parameters else 1
            self.batch_size = train_hyper_parameters['batch_size'] if 'batch_size' in train_hyper_parameters else 1
            self.is_eval = train_hyper_parameters['is_eval'] if 'is_eval' in train_hyper_parameters else False
        else:
            self.epochs = 1
            self.batch_size = 1
            self.is_eval = False

    def _parse_train_net(self):
        """Parse information on training net."""
        if 'train_net' not in self.model_data:
            raise ValueError('FLYamlData init failed: missing field of \'train_net\'')
        self.train_net = self.model_data['train_net']
        if 'inputs' not in self.train_net:
            raise ValueError('FLYamlData init failed: missing field of \'inputs\' of \'train_net\'')
        self.train_net_ins = [input_config for input_config in self.train_net['inputs']]
        if not self.train_net_ins:
            raise ValueError('FLYamlData init failed: inputs of \'train_net\' are empty')
        self.train_net_in_names = [input_config['name'] for input_config in self.train_net['inputs']]
        if 'outputs' not in self.train_net:
            raise ValueError('FLYamlData init failed: missing field of \'outputs\' of \'train_net\'')
        self.train_net_outs = [output_config for output_config in self.train_net['outputs']]
        if not self.train_net_outs:
            raise ValueError('FLYamlData init failed: outputs of \'train_net\' are empty')
        self.train_net_out_names = [input_config['name'] for input_config in self.train_net['outputs']]

    def _parse_eval_net(self):
        """Parse information on evaluation net."""
        if 'eval_net' not in self.model_data:
            raise ValueError('FLYamlData init failed: missing field of \'eval_net\'')
        self.eval_net = self.model_data['eval_net']
        if 'inputs' not in self.eval_net:
            raise ValueError('FLYamlData init failed: missing field of \'inputs\' of \'eval_net\'')
        self.eval_net_ins = [input_config for input_config in self.eval_net['inputs']]
        if not self.eval_net_ins:
            raise ValueError('FLYamlData init failed: inputs of \'eval_net\' are empty')
        if 'outputs' not in self.eval_net:
            raise ValueError('FLYamlData init failed: missing field of \'outputs\' of \'eval_net\'')
        self.eval_net_outs = [output_config for output_config in self.eval_net['outputs']]
        if not self.eval_net_outs:
            raise ValueError('FLYamlData init failed: outputs of \'eval_net\' are empty')
        self.eval_net_gt = self.eval_net['gt'] if 'gt' in self.eval_net else None

    def _check_eps(self, privacy, dp_name='embedding_dp'):
        """validate the eps parameter in dp mechanism"""
        if not privacy[dp_name] or 'eps' not in privacy[dp_name]:
            raise ValueError(f'FLYamlData init failed: parameter eps missed for {dp_name}.')
        eps = privacy[dp_name]['eps']
        if eps:
            if not isinstance(eps, (int, float)):
                raise TypeError(f'FLYamlData init failed: parameter eps must be an int or a float number, \
                                but {type(eps)} found.')
            if eps < 0:
                raise ValueError(f'FLYamlData init failed: parameter eps cannot be negative, but got {eps}.')
        return eps

    def _parse_privacy(self):
        """Verify configurations of privacy defined in the yaml file."""
        if 'privacy' in self.yaml_data:
            privacy = self.yaml_data['privacy']
            if privacy:
                if 'embedding_dp' in privacy:
                    self.embedding_dp_eps = self._check_eps(privacy, 'embedding_dp')
                if 'label_dp' in privacy:
                    eps = self._check_eps(privacy, 'label_dp')
                    if eps or eps == 0:
                        self.label_dp_eps = eps
                    else:
                        raise ValueError(f'FLYamlData init failed: the value of eps is missing.')
                for scheme in privacy.keys():
                    if scheme not in ('embedding_dp', 'label_dp'):
                        raise ValueError(f'FLYamlData init failed: unknown privacy scheme {scheme}.')

    def _check_opts(self):
        """Verify configurations of optimizers defined in the yaml file."""
        for opt_config in self.opts:
            for grad_config in opt_config['grads']:
                grad_inputs = {grad_in['name'] for grad_in in grad_config['inputs']}
                if not grad_inputs.issubset(self.train_net_in_names):
                    raise ValueError('optimizer %s config error: contains undefined inputs' % opt_config['type'])
                grad_out = grad_config['output']['name']
                if grad_out not in self.train_net_out_names:
                    raise ValueError('optimizer %s config error: contains undefined output %s'
                                     % (opt_config['type'], grad_out))
                if 'sens' in grad_config and not isinstance(grad_config['sens'], (str, int, float)):
                    raise ValueError('optimizer %s config error: unsupported sens type of grads' % opt_config['type'])

    def _check_grad_scalers(self):
        """Verify configurations of grad_scales defined in the yaml file."""
        for grad_scale_config in self.grad_scalers:
            if 'inputs' not in grad_scale_config or not grad_scale_config['inputs']:
                raise ValueError('FLYamlData init failed：\'grad_scalers\' contains no inputs')
            grad_scale_ins = {grad_scale_in['name'] for grad_scale_in in grad_scale_config['inputs']}
            if not grad_scale_ins.issubset(self.train_net_in_names):
                raise ValueError('FLYamlData init failed: \'grad_scalers\' contains undefined inputs')
            if 'output' not in grad_scale_config:
                raise ValueError('FLYamlData init failed：\'grad_scalers\' contains no output')
            grad_scale_out = grad_scale_config['output']['name']
            if grad_scale_out not in self.train_net_out_names:
                raise ValueError('FLYamlData init failed: \'grad_scalers\' contains undefined output %s'
                                 % grad_scale_out)


def get_params_list_by_name(net, name):
    """
    Get parameters list by name from the nn.Cell

    Inputs:
        net (nn.Cell): Network described using mindspore.
        name (str): Name of parameters to be gotten.
    """
    res = []
    trainable_params = net.trainable_params()
    for param in trainable_params:
        if name in param.name:
            res.append(param)
    return res


def get_params_by_name(net, weight_name_list):
    """
    Get parameters list by names from the nn.Cell

    Inputs:
        net (nn.Cell): Network described using mindspore.
        name (list): Names of parameters to be gotten.
    """
    params = []
    for weight_name in weight_name_list:
        params.extend(get_params_list_by_name(net, weight_name))
    params = ParameterTuple(params)
    return params


class IthOutputCellInDict(nn.Cell):
    """
    Encapulate network with multiple outputs so that it only output one variable.

    Args:
        network (nn.Cell): Network to be encapulated.
        output_index (int): Index of the output variable.

    Inputs:
        **kwargs (dict): input of the network.
    """

    def __init__(self, network, output_index):
        super(IthOutputCellInDict, self).__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, **kwargs):
        return self.network(**kwargs)[self.output_index]


class IthOutputCellInTuple(nn.Cell):
    """
    Encapulate network with multiple outputs so that it only output one variable.

    Args:
        network (nn.Cell): Network to be encapulated.
        output_index (int): Index of the output variable.

    Inputs:
        *kwargs (tuple): input of the network.
    """

    def __init__(self, network, output_index):
        super(IthOutputCellInTuple, self).__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, *args):
        return self.network(*args)[self.output_index]
