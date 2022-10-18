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
"""Test the functions of FLModel"""
import os
import pytest

import mindspore as ms
from mindspore import context
import mindspore.nn as nn
from mindspore.common.initializer import Normal

from mindspore_federated import FLModel, FLYamlData


class LeaderNet(nn.Cell):
    """Leader network on the basis of LeNet-5"""

    def __init__(self, num_class=10):
        super(LeaderNet, self).__init__()
        self.fc3 = nn.Dense(84, num_class)

    def construct(self, embedding):
        """construct LeaderNet"""
        logits = self.fc3(embedding)
        return logits


class FollowerNet(nn.Cell):
    """Follower network on the basis of LeNet-5"""

    def __init__(self, num_channel=1):
        super(FollowerNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        """construct FollowerNet"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        embedding = self.relu(x)
        return embedding


class LeaderTrainEvalNet(nn.Cell):
    """Net of Leader for training and eval."""

    def __init__(self, net):
        super(LeaderTrainEvalNet, self).__init__()
        self.net = net
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)

    def construct(self, embedding, label):
        """construct LeaderTrainEvalNet"""
        logits = self.net(embedding)
        loss_value = self.loss(logits, label)
        return loss_value


@pytest.mark.skipif(ms.__version__ < '1.8.1', reason='rely on mindspore >= 1.8.1')
def test_fl_model_basic():
    """
    Feature: Basic functions of FLModel, including constructing models for FL, controlling forward & backward
        processes, etc.
    Description: Construct a simple vertical FL system, and test whether it works normally.
    Expectation: Success
    """
    context.set_context(mode=context.GRAPH_MODE)
    leader_base_net = LeaderNet(10)
    leader_train_net = leader_eval_net = LeaderTrainEvalNet(leader_base_net)
    leader_yaml = FLYamlData(os.path.join(os.getcwd(), 'yaml_files/leader.yaml'))

    follower_train_net = follower_eval_net = follower_base_net = FollowerNet(1)
    follower_yaml = FLYamlData(os.path.join(os.getcwd(), 'yaml_files/follower.yaml'))

    eval_metric = nn.Accuracy()

    leader_fl_model = FLModel(yaml_data=leader_yaml,
                              network=leader_base_net,
                              train_network=leader_train_net,
                              eval_network=leader_eval_net,
                              metrics=eval_metric)
    follower_fl_model = FLModel(yaml_data=follower_yaml,
                                network=follower_base_net,
                                train_network=follower_train_net,
                                eval_network=follower_eval_net)
    # construct random data
    feature = ms.Tensor(shape=(1, 1, 32, 32), dtype=ms.float32, init=Normal())
    label = ms.Tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=ms.float32)
    leader_local_data = {'label': label}
    follower_local_data = {'x': feature}
    # forward process
    follower_out = follower_fl_model.forward_one_step(follower_local_data)
    leader_fl_model.forward_one_step(leader_local_data, follower_out)
    # backward process
    scale = leader_fl_model.backward_one_step(leader_local_data, follower_out)
    follower_fl_model.backward_one_step(follower_local_data, sens=scale)


def test_invalid_basic_network():
    """
    Feature: Raise ValueError when the developer inputs invalid basic networks
    Description: Construct a simple vertical FL system, then input invalid basic networks.
    Expectation: Raise TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    # test case #1: invalid basic network
    base_net = [1, 2, 3, 4]
    train_net = eval_net = LeaderTrainEvalNet(LeaderNet(10))
    yaml_data = FLYamlData(os.path.join(os.getcwd(), 'yaml_files/leader.yaml'))
    with pytest.raises(TypeError):
        FLModel(yaml_data, base_net, train_network=train_net, eval_network=eval_net)


def test_invalid_train_network():
    """
    Feature: Raise ValueError when the developer inputs invalid train networks
    Description: Construct a simple vertical FL system, then input invalid train networks.
    Expectation: Raise TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    base_net = LeaderNet(10)
    train_net = {'x': ms.Tensor([[1, 2], [3, 4]], dtype=ms.float32)}
    eval_net = LeaderTrainEvalNet(base_net)
    yaml_data = FLYamlData(os.path.join(os.getcwd(), 'yaml_files/leader.yaml'))
    with pytest.raises(TypeError):
        FLModel(yaml_data, base_net, train_network=train_net, eval_network=eval_net)


def test_invalid_eval_network():
    """
    Feature: Raise ValueError when the developer inputs invalid eval networks
    Description: Construct a simple vertical FL system, then input invalid eval networks.
    Expectation: Raise TypeError.
    """
    base_net = LeaderNet(10)
    train_net = LeaderTrainEvalNet(base_net)
    eval_net = {'x': ms.Tensor([[1, 2], [3, 4]], dtype=ms.float32)}
    yaml_data = FLYamlData(os.path.join(os.getcwd(), 'yaml_files/leader.yaml'))
    with pytest.raises(TypeError):
        FLModel(yaml_data, base_net, train_network=train_net, eval_network=eval_net)


@pytest.mark.skipif(ms.__version__ < '1.8.1', reason='rely on mindspore >= 1.8.1')
def test_loss_fn():
    """
    Feature: Construct training network with loss_ln and basic network.
    Description: Input a basic network and loss_ln, FLModel will construct the training network.
    Expectation: Success
    """
    context.set_context(mode=context.GRAPH_MODE)
    leader_base_net = LeaderNet(10)
    leader_yaml = FLYamlData(os.path.join(os.getcwd(), 'yaml_files/leader.yaml'))
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    leader_fl_model = FLModel(leader_yaml, leader_base_net, loss_fn=loss_fn)

    follower_train_net = follower_eval_net = follower_base_net = FollowerNet(1)
    follower_yaml = FLYamlData(os.path.join(os.getcwd(), 'yaml_files/follower.yaml'))
    follower_fl_model = FLModel(yaml_data=follower_yaml,
                                network=follower_base_net,
                                train_network=follower_train_net,
                                eval_network=follower_eval_net)

    feature = ms.Tensor(shape=(1, 1, 32, 32), dtype=ms.float32, init=Normal())
    label = ms.Tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=ms.float32)
    leader_local_data = {'label': label}
    follower_local_data = {'x': feature}
    follower_out = follower_fl_model.forward_one_step(follower_local_data)
    leader_fl_model.forward_one_step(leader_local_data, follower_out)
    scale = leader_fl_model.backward_one_step(leader_local_data, follower_out)
    follower_fl_model.backward_one_step(follower_local_data, sens=scale)


def test_no_optim():
    """
    Feature: Raise AttributeError when not specify training network nor loss_fn.
    Description: Input no training network nor loss_fn.
    Expectation: Raise AttributeError
    """
    context.set_context(mode=context.GRAPH_MODE)
    leader_base_net = LeaderNet(10)
    leader_yaml = FLYamlData(os.path.join(os.getcwd(), 'yaml_files/leader.yaml'))
    with pytest.raises(AttributeError):
        FLModel(leader_yaml, leader_base_net)


def test_customized_opts():
    """
    Feature: Construct customized optimizers for training network.
    Description: Input a customized optimizer as args of FLModel.
    Expectation: Success
    """
    context.set_context(mode=context.GRAPH_MODE)
    # test case #1: input a customized optimizer
    leader_base_net = LeaderNet(10)
    leader_train_net = LeaderTrainEvalNet(leader_base_net)
    leader_yaml = FLYamlData(os.path.join(os.getcwd(), 'yaml_files/leader.yaml'))

    class CustomizedOptim():
        """demo of customized optimizer"""

        def __call__(self, logit, gt):
            if isinstance(logit, ms.Tensor) and isinstance(gt, ms.Tensor):
                return True
            return False

    optim = CustomizedOptim()
    leader_fl_model = FLModel(yaml_data=leader_yaml,
                              network=leader_base_net,
                              train_network=leader_train_net,
                              optimizers=optim)
    label = ms.Tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=ms.float32)
    leader_local_data = {'label': label}
    embedding = ms.Tensor(shape=(1, 84), dtype=ms.float32, init=Normal())
    leader_remote_data = {'embedding': embedding}
    leader_fl_model.forward_one_step(leader_local_data, leader_remote_data)
    leader_fl_model.backward_one_step(leader_local_data, leader_remote_data)
