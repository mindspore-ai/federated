# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Test the functions of communication compression"""
import pytest
from mindspore_federated.startup.compress_config import CompressConfig
from common import communication_compression_test


@communication_compression_test
def test_vfl_communication_compression_compress_type():
    """
    Feature: Test data join: compress_type is wrong.
    Description: No input.
    Expectation: ERROR log is right and success.
    """
    with pytest.raises(ValueError) as err:
        CompressConfig(compress_type="james", bit_num=8)
    err_str = str(err.value)
    assert_msg = "compress_type" in err_str and "str" in err_str
    assert assert_msg


@communication_compression_test
def test_vfl_communication_compression_bit_num():
    """
    Feature: Test data join: bit_num is wrong.
    Description: No input.
    Expectation: ERROR log is right and success.
    """
    with pytest.raises(ValueError) as err:
        CompressConfig(compress_type="min_max", bit_num=23)
    err_str = str(err.value)
    assert_msg = "bit_num" in err_str and "range of" in err_str
    assert assert_msg
