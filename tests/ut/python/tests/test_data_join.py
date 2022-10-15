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
from multiprocessing import Process
import pytest
import pandas as pd
from mindspore_federated.data_join import FLDataWorker
from mindspore_federated.data_join import load_mindrecord
from common import vfl_data_test
from common_data import generate_worker_config, generate_schema


@vfl_data_test
def test_case_data_join_role():
    """
    Feature: Test data join: role is wrong
    Description: Input constructed through generate_random_data
    Expectation: ERROR log is right and success.
    """
    role = "server"
    worker_config_path = "temp/leader.yaml"
    schema_path = "temp/schema.yaml"
    generate_worker_config(
        role="leader",
        file_num=4,
        primary_key="oaid",
        bucket_num=5,
        store_type="csv",
        shard_num=1,
        join_type="psi",
        thread_num=0,
    )
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
    server_address = "127.0.0.1:6969"
    peer_server_address = "127.0.0.1:9696"
    with pytest.raises(ValueError) as err:
        FLDataWorker(role=role,
                     worker_config_path=worker_config_path,
                     data_schema_path=schema_path,
                     server_address=server_address,
                     peer_server_address=peer_server_address,
                     )
    assert "role must be \"leader\" or \"follower\"" in str(err.value)


@vfl_data_test
def test_case_data_join_join_type():
    """
    Feature: Test data join: join_type is wrong
    Description: Input constructed through generate_random_data
    Expectation: ERROR log is right and success.
    """
    role = "leader"
    worker_config_path = "temp/leader.yaml"
    schema_path = "temp/schema.yaml"
    generate_worker_config(
        role="leader",
        file_num=4,
        primary_key="oaid",
        bucket_num=5,
        store_type="csv",
        shard_num=1,
        join_type="wtc",
        thread_num=0,
    )
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
    server_address = "127.0.0.1:6969"
    peer_server_address = "127.0.0.1:9696"
    with pytest.raises(ValueError) as err:
        FLDataWorker(role=role,
                     worker_config_path=worker_config_path,
                     data_schema_path=schema_path,
                     server_address=server_address,
                     peer_server_address=peer_server_address,
                     )
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
    role = "leader"
    worker_config_path = "temp/leader.yaml"
    schema_path = "temp/schema.yaml"
    generate_worker_config(
        role="leader",
        file_num=4,
        primary_key="oaid",
        bucket_num=0,
        store_type="csv",
        shard_num=1,
        join_type="psi",
        thread_num=0,
    )
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
    server_address = "127.0.0.1:6969"
    peer_server_address = "127.0.0.1:9696"
    with pytest.raises(ValueError) as err:
        FLDataWorker(role=role,
                     worker_config_path=worker_config_path,
                     data_schema_path=schema_path,
                     server_address=server_address,
                     peer_server_address=peer_server_address,
                     )
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
    role = "leader"
    worker_config_path = "temp/leader.yaml"
    schema_path = "temp/schema.yaml"
    generate_worker_config(
        role="leader",
        file_num=4,
        primary_key="oaid",
        bucket_num=1000001,
        store_type="csv",
        shard_num=1,
        join_type="psi",
        thread_num=0,
    )
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
    server_address = "127.0.0.1:6969"
    peer_server_address = "127.0.0.1:9696"
    with pytest.raises(ValueError) as err:
        FLDataWorker(role=role,
                     worker_config_path=worker_config_path,
                     data_schema_path=schema_path,
                     server_address=server_address,
                     peer_server_address=peer_server_address,
                     )
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
    role = "leader"
    worker_config_path = "temp/leader.yaml"
    schema_path = "temp/schema.yaml"
    generate_worker_config(
        role="leader",
        file_num=4,
        primary_key="oaid",
        bucket_num=5,
        store_type="csv",
        shard_num=0,
        join_type="psi",
        thread_num=0,
    )
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
    server_address = "127.0.0.1:6969"
    peer_server_address = "127.0.0.1:9696"
    with pytest.raises(ValueError) as err:
        FLDataWorker(role=role,
                     worker_config_path=worker_config_path,
                     data_schema_path=schema_path,
                     server_address=server_address,
                     peer_server_address=peer_server_address,
                     )
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
    role = "leader"
    worker_config_path = "temp/leader.yaml"
    schema_path = "temp/schema.yaml"
    generate_worker_config(
        role="leader",
        file_num=4,
        primary_key="oaid",
        bucket_num=5,
        store_type="csv",
        shard_num=1001,
        join_type="psi",
        thread_num=0,
    )
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
    server_address = "127.0.0.1:6969"
    peer_server_address = "127.0.0.1:9696"
    with pytest.raises(ValueError) as err:
        FLDataWorker(role=role,
                     worker_config_path=worker_config_path,
                     data_schema_path=schema_path,
                     server_address=server_address,
                     peer_server_address=peer_server_address,
                     )
    err_str = str(err.value)
    assert_msg = "shard_num" in err_str and "[1, 1000]" in err_str
    assert assert_msg


def worker_process_fun(
        role="leader",
        server_address="127.0.0.1:6969",
        peer_server_address="127.0.0.1:9696",
):
    """start vfl data worker"""
    worker = FLDataWorker(
        role=role,
        worker_config_path="temp/{}.yaml".format(role),
        data_schema_path="temp/{}_schema.yaml".format(role),
        server_address=server_address,
        peer_server_address=peer_server_address,
    )
    worker.export()


@vfl_data_test
def test_case_data_join_demo():
    """
    Feature: Test data join: bucket_num is too big
    Description: Input constructed through generate_random_data
    Expectation: ERROR log is right and success.
    """
    generate_worker_config(
        role="leader",
        file_num=4,
        primary_key="oaid",
        bucket_num=5,
        store_type="csv",
        shard_num=1,
        join_type="psi",
        thread_num=0,
    )
    generate_schema(
        yaml_path="temp/leader_schema.yaml",
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

    leader_process = Process(target=worker_process_fun, args=("leader", "127.0.0.1:6969", "127.0.0.1:9696"))
    leader_process.start()

    generate_worker_config(
        role="follower",
        file_num=2,
        primary_key="oaid",
        bucket_num=5,
        store_type="csv",
        shard_num=1,
        join_type="psi",
        thread_num=0,
    )
    generate_schema(
        yaml_path="temp/follower_schema.yaml",
        oaid="string",
        feature10="float32",
        feature11="float32",
        feature12="float32",
        feature13="float32",
        feature14="float32",
        feature15="float32",
        feature16="float32",
        feature17="float32",
        feature18="float32",
        feature19="float32",
    )
    follower_process = Process(target=worker_process_fun, args=("follower", "127.0.0.1:9696", "127.0.0.1:6969"))
    follower_process.start()

    leader_process.join(timeout=30)
    follower_process.join(timeout=30)

    # verify joined data with real intersection data
    leader_oaid_list = list()
    follower_oaid_list = list()
    dataset = load_mindrecord(input_dir="temp/leader", shuffle=True, seed=0)
    for key in dataset.create_dict_iterator():
        leader_oaid_list.append(key["oaid"].asnumpy())
    dataset = load_mindrecord(input_dir="temp/follower", shuffle=True, seed=0)
    for key in dataset.create_dict_iterator():
        follower_oaid_list.append(key["oaid"].asnumpy())

    real_intersection = pd.read_csv("temp/intersection_data.csv", usecols=["oaid"]).values[:, 0]
    assert len(leader_oaid_list) == len(follower_oaid_list) == len(real_intersection)
    for leader_oaid, follower_oaid in zip(leader_oaid_list, follower_oaid_list):
        assert leader_oaid == follower_oaid
        assert leader_oaid in real_intersection
