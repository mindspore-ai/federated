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
"""Test the Label Differential Privacy feature"""
import os
import pytest
import numpy as np

import mindspore
from mindspore import Tensor, context, nn
from mindspore.common.initializer import Normal
from mindspore_federated import FLModel, FLYamlData
from mindspore_federated.privacy import LabelDP
from test_fl_model import LeaderNet, LeaderTrainEvalNet, FollowerNet


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_labeldp_binary_labels(mode):
    """
    Feature: Convert input binary labels into dp labels under given privacy parameter eps.
    Description: Input a tensor of binary labels, output dp labels of the same type and shape.
    Expectation: Success
    """
    context.set_context(mode=mode)
    for eps in [0, 0.0, 1, 1.0, 10, 10.0]:
        label_dp = LabelDP(eps=eps)
        label = Tensor(np.zeros((5, 1)), dtype=mindspore.float32)
        label_dp(label)
        label = Tensor(np.zeros(5), dtype=mindspore.float32)
        label_dp(label)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_labeldp_onehot_labels(mode):
    """
    Feature: Convert input onehot labels into dp labels under given privacy parameter eps.
    Description: Input a tensor of onehot labels, output dp labels of the same type and shape.
    Expectation: Success
    """
    context.set_context(mode=mode)
    for eps in [0, 0.0, 1, 1.0, 10, 10.0]:
        label_dp = LabelDP(eps=eps)
        label = Tensor(np.hstack((np.ones((5, 1)), np.zeros((5, 2)))), dtype=mindspore.float32)
        label_dp(label)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_labeldp_invalid_eps_type(mode):
    """
    Feature: Except TypeError when the type of eps is invalid.
    Description: Create a LabelDP object with an eps of invalid type.
    Expectation: Raise TypeError
    """
    context.set_context(mode=mode)
    with pytest.raises(TypeError):
        LabelDP('1.0')


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_labeldp_invalid_eps_value(mode):
    """
    Feature: Except ValueError when eps is a negative value.
    Description: Create a LabelDP object with a negative eps value.
    Expectation: Raise ValueError
    """
    context.set_context(mode=mode)
    with pytest.raises(ValueError):
        LabelDP(-1.0)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_labeldp_invalid_label_type(mode):
    """
    Feature: Except TypeError when the type of input labels is invalid.
    Description: Input a batch of labels of an invalid type.
    Expectation: Raise TypeError
    """
    context.set_context(mode=mode)
    with pytest.raises(TypeError):
        label_dp = LabelDP(1.0)
        label = np.zeros(5)
        label_dp(label)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_labeldp_invalid_label_value(mode):
    """
    Feature: Except ValueError when the input labels are not binary or onehot.
    Description: Input a batch of labels of invalid values.
    Expectation: Raise ValueError
    """
    context.set_context(mode=mode)
    for label in [Tensor([0, 1, 2]), Tensor([[1, 0, 0], [1, 1, 0]]), Tensor(np.zeros((2, 2, 1)))]:
        with pytest.raises(ValueError):
            label_dp = LabelDP(1.0)
            label_dp(label)


@pytest.mark.skipif(mindspore.__version__ < '1.8.1', reason='reply on mindspore >= 1.8.1')
def test_labeldp_in_fl_model():
    """
    Feature: Run label dp in FLModel.
    Description: Config privacy.label_dp module in the yaml file, to enable label dp in FLModel.
    Expectation: Success
    """
    context.set_context(mode=context.GRAPH_MODE)
    leader_base_net = LeaderNet(10)
    leader_train_net = leader_eval_net = LeaderTrainEvalNet(leader_base_net)
    leader_yaml = FLYamlData(os.path.join(os.getcwd(), 'yaml_files/leader_label_dp.yaml'))
    follower_train_net = follower_eval_net = FollowerNet(1)
    follower_yaml = FLYamlData(os.path.join(os.getcwd(), 'yaml_files/follower.yaml'))
    eval_metric = nn.Accuracy()

    leader_fl_model = FLModel(yaml_data=leader_yaml,
                              network=leader_train_net,
                              eval_network=leader_eval_net,
                              metrics=eval_metric)
    follower_fl_model = FLModel(yaml_data=follower_yaml,
                                network=follower_train_net,
                                eval_network=follower_eval_net)
    # construct random data
    feature = Tensor(shape=(1, 1, 32, 32), dtype=mindspore.float32, init=Normal())
    label = Tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=mindspore.float32)
    leader_local_data = {'label': label}
    follower_local_data = {'x': feature}
    # forward process
    follower_out = follower_fl_model.forward_one_step(follower_local_data)
    leader_fl_model.forward_one_step(leader_local_data, follower_out)
    # backward process
    scale = leader_fl_model.backward_one_step(leader_local_data, follower_out)
    follower_fl_model.backward_one_step(follower_local_data, sens=scale)
