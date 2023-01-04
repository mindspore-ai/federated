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
"""classes of TEE mechanisms for the federated learner."""
from mindspore import ParameterTuple, nn, ops
from mindspore.nn.optim import SGD


class Linear(nn.Cell):
    """Network used in SimuTEE"""
    def __init__(self, total_len):
        super().__init__()
        self.dense = nn.Dense(total_len, total_len)
        self.concat = ops.Concat(axis=1)

    def construct(self, *input_features):
        x = self.concat(input_features)
        x = self.dense(x)
        return x


class SimuTEE():
    """class for the simulated TEE layer"""
    def __init__(self, yaml_data):
        super().__init__()
        self.network = Linear(yaml_data['input_dim'])
        self.grad_op_param = ops.GradOperation(get_by_list=True, sens_param=True)
        self.grad_op_input = ops.GradOperation(get_all=True, sens_param=True)
        self.params = ParameterTuple(self.network.trainable_params())
        opt_config = yaml_data['opt']
        self.optimizer = SGD(self.params, opt_config['learning_rate'], opt_config['momentum'], opt_config['loss_scale'])
        self.concat = ops.Concat(axis=1)
        self.input_data_batch = None

    def forward_one_step(self, *input_data_batch):
        """forward one step during training"""
        self.input_data_batch = input_data_batch
        out = self.network(*input_data_batch)

        feature_lens = [len(feature[0]) for feature in input_data_batch]
        out_data_batch = []

        for i in range(len(feature_lens)):
            out_data_batch.append(out[:, sum(feature_lens[:i]):sum(feature_lens[:i + 1])])
        return tuple(out_data_batch)

    def backward_one_step(self, *input_grad_batch):
        """backward one step during training"""
        grad_sens = self.concat(list(input_grad_batch))
        grad_input = self.grad_op_input(self.network)(*self.input_data_batch, grad_sens)
        grad_param = self.grad_op_param(self.network, self.params)(*self.input_data_batch, grad_sens)
        self.optimizer(grad_param)
        return grad_input
