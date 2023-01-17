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
"""Test the functions of DataJoinDemo"""
import os
import pytest
from mindspore_federated import FLDataWorker
from mindspore_federated.common.config import parse_yaml
from common import vfl_data_test
from common_data import generate_schema


@vfl_data_test
def test_case_data_join_role():
    """
    Feature: Test data join: role is wrong
    Description: Input constructed through generate_random_data
    Expectation: ERROR log is right and success.
    """
    generate_schema(
        yaml_path="temp/schema.yaml",
        oaid="string",
        feature0="float32",
        feature1="float32",
        feature2="float32",
        feature3="float32",
        feature4="float32",
        feature5="float32",
        feature6="float32",
        feature7="float32",
        feature8="float32",
        feature9="float32",
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_dict, _, _ = parse_yaml(os.path.join(current_dir, "yaml_files/vfl_data_join_config.yaml"))
    cfg_dict['role'] = "wtc"
    with pytest.raises(ValueError) as err:
        work = FLDataWorker(cfg_dict)
        work.do_worker()
    assert "role must be \"leader\" or \"follower\"" in str(err.value)


@vfl_data_test
def test_case_data_join_join_type():
    """
    Feature: Test data join: join_type is wrong
    Description: Input constructed through generate_random_data
    Expectation: ERROR log is right and success.
    """
    generate_schema(
        yaml_path="temp/schema.yaml",
        oaid="string",
        feature0="float32",
        feature1="float32",
        feature2="float32",
        feature3="float32",
        feature4="float32",
        feature5="float32",
        feature6="float32",
        feature7="float32",
        feature8="float32",
        feature9="float32",
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_dict, _, _ = parse_yaml(os.path.join(current_dir, "yaml_files/vfl_data_join_config.yaml"))
    cfg_dict['join_type'] = "wtc"
    with pytest.raises(ValueError) as err:
        work = FLDataWorker(cfg_dict)
        work.do_worker()
    err_str = str(err.value)
    assert_msg = "join_type" in err_str and "str" in err_str
    assert assert_msg


@vfl_data_test
def test_case_data_join_small_bucket_num():
    """
    Feature: Test data join: bucket_num is too small
    Description: Input constructed through generate_random_data
    Expectation: ERROR log is right and success.
    """
    generate_schema(
        yaml_path="temp/schema.yaml",
        oaid="string",
        feature0="float32",
        feature1="float32",
        feature2="float32",
        feature3="float32",
        feature4="float32",
        feature5="float32",
        feature6="float32",
        feature7="float32",
        feature8="float32",
        feature9="float32",
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_dict, _, _ = parse_yaml(os.path.join(current_dir, "yaml_files/vfl_data_join_config.yaml"))
    cfg_dict['bucket_num'] = 0
    with pytest.raises(ValueError) as err:
        work = FLDataWorker(cfg_dict)
        work.do_worker()
    err_str = str(err.value)
    assert_msg = "bucket_num" in err_str and "[1, 1000000]" in err_str
    assert assert_msg


@vfl_data_test
def test_case_data_join_big_bucket_num():
    """
    Feature: Test data join: bucket_num is too big
    Description: Input constructed through generate_random_data
    Expectation: ERROR log is right and success.
    """
    generate_schema(
        yaml_path="temp/schema.yaml",
        oaid="string",
        feature0="float32",
        feature1="float32",
        feature2="float32",
        feature3="float32",
        feature4="float32",
        feature5="float32",
        feature6="float32",
        feature7="float32",
        feature8="float32",
        feature9="float32",
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_dict, _, _ = parse_yaml(os.path.join(current_dir, "yaml_files/vfl_data_join_config.yaml"))
    cfg_dict['bucket_num'] = 1000001
    with pytest.raises(ValueError) as err:
        work = FLDataWorker(cfg_dict)
        work.do_worker()
    err_str = str(err.value)
    assert_msg = "bucket_num" in err_str and "[1, 1000000]" in err_str
    assert assert_msg


@vfl_data_test
def test_case_data_join_small_shard_num():
    """
    Feature: Test data join: shard_num is too small
    Description: Input constructed through generate_random_data
    Expectation: ERROR log is right and success.
    """
    generate_schema(
        yaml_path="temp/schema.yaml",
        oaid="string",
        feature0="float32",
        feature1="float32",
        feature2="float32",
        feature3="float32",
        feature4="float32",
        feature5="float32",
        feature6="float32",
        feature7="float32",
        feature8="float32",
        feature9="float32",
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_dict, _, _ = parse_yaml(os.path.join(current_dir, "yaml_files/vfl_data_join_config.yaml"))
    cfg_dict['shard_num'] = 0
    with pytest.raises(ValueError) as err:
        work = FLDataWorker(cfg_dict)
        work.do_worker()
    err_str = str(err.value)
    assert_msg = "shard_num" in err_str and "[1, 1000]" in err_str
    assert assert_msg


@vfl_data_test
def test_case_data_join_big_shard_num():
    """
    Feature: Test data join: shard_num is too big
    Description: Input constructed through generate_random_data
    Expectation: ERROR log is right and success.
    """
    generate_schema(
        yaml_path="temp/schema.yaml",
        oaid="string",
        feature0="float32",
        feature1="float32",
        feature2="float32",
        feature3="float32",
        feature4="float32",
        feature5="float32",
        feature6="float32",
        feature7="float32",
        feature8="float32",
        feature9="float32",
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_dict, _, _ = parse_yaml(os.path.join(current_dir, "yaml_files/vfl_data_join_config.yaml"))
    cfg_dict['shard_num'] = 1001
    with pytest.raises(ValueError) as err:
        work = FLDataWorker(cfg_dict)
        work.do_worker()
    err_str = str(err.value)
    assert_msg = "shard_num" in err_str and "[1, 1000]" in err_str
    assert assert_msg
