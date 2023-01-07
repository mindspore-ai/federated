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
import pytest
from mindspore import Tensor, context
from mindspore_federated.privacy import EmbeddingDP


MODE = [context.GRAPH_MODE, context.PYNATIVE_MODE]


@pytest.mark.parametrize('mode', MODE)
@pytest.mark.parametrize('eps', ['1', (1,), [1], {1: 1}])
def test_embeddingdp_invalid_eps_type(mode, eps):
    """
    Feature: Except TypeError when eps is neither an int nor a float.
    Description: Create an EmbeddingDP object with invalid types.
    Expectation: Raise TypeError
    """
    context.set_context(mode=mode)
    with pytest.raises(TypeError):
        EmbeddingDP(eps)


@pytest.mark.parametrize('mode', MODE)
def test_embeddingdp_invalid_eps_value(mode):
    """
    Feature: Except ValueError when eps less than 0.
    Description: Create an EmbeddingDP object with invalid values.
    Expectation: Raise ValueError
    """
    context.set_context(mode=mode)
    with pytest.raises(ValueError):
        EmbeddingDP(-1)


@pytest.mark.parametrize('mode', MODE)
@pytest.mark.parametrize('inputs', [None, '1', (1,), [1], True, {1: 1}])
def test_embeddingdp_invalid_inputs_type(mode, inputs):
    """
    Feature: Except TypeError when the input is not a tensor.
    Description: Create an EmbeddingDP object and feed it with invalid inputs.
    Expectation: Raise TypeError
    """
    context.set_context(mode=mode)
    with pytest.raises(TypeError):
        embedding_dp = EmbeddingDP(1.0)
        embedding_dp(inputs)


@pytest.mark.parametrize('mode', MODE)
def test_embeddingdp_unary_encoding(mode):
    """
    Feature: Test the functionality of unary encoding.
    Description: Create an EmbeddingDP object without eps.
    Expectation: The values are converted to 1 when they are larger than 0, otherwise to 0
    """
    context.set_context(mode=mode)
    inputs = Tensor([-4, 1.3, -0.02, 32, 0])
    expected = Tensor([0, 1, 0, 1, 0])
    embedding_dp = EmbeddingDP()
    outputs = embedding_dp(inputs)
    assert all(outputs == expected)
