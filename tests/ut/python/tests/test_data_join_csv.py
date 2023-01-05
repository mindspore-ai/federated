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
import os
import pandas as pd
from mindspore_federated import FLDataWorker
from mindspore_federated.data_join import load_mindrecord
from mindspore_federated.common.config import parse_yaml
from common import vfl_data_test, get_default_ssl_config
from common_data import generate_schema


def worker_process_fun(
        role="leader",
        server_name="server",
        target_server_name="client",
        http_server_address="127.0.0.1:6969",
        remote_server_address="127.0.0.1:9696"
):
    """start vfl data worker"""
    file_num = 4 if role == "leader" else 2
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_cert_path, client_cert_path, ca_cert_path, _, _ = get_default_ssl_config()

    cfg_dict, _, _ = parse_yaml(os.path.join(current_dir, "yaml_files/vfl_data_join_config.yaml"))
    cfg_dict['server_name'] = server_name
    cfg_dict['http_server_address'] = http_server_address
    cfg_dict['remote_server_name'] = target_server_name
    cfg_dict['remote_server_address'] = remote_server_address
    cfg_dict['server_cert_path'] = server_cert_path
    cfg_dict['client_cert_path'] = client_cert_path
    cfg_dict['ca_cert_path'] = ca_cert_path
    cfg_dict['enable_ssl'] = True
    cfg_dict['main_table_files'] = ["temp/{}_data_{}.csv".format(role, _) for _ in range(file_num)]
    cfg_dict['role'] = role
    cfg_dict['output_dir'] = "temp/{}/".format(role)
    cfg_dict['data_schema_path'] = "temp/{}_schema.yaml".format(role)
    cfg_dict['shard_num'] = 2
    work = FLDataWorker(cfg_dict)
    work.do_worker()


@vfl_data_test
def test_case_data_join_demo():
    """
    Feature: Test data join: whole flow.
    Description: Input constructed through generate_random_data
    Expectation: ERROR log is right and success.
    """
    generate_schema(
        yaml_path="temp/leader_schema.yaml",
        oaid="string",
        feature0="int32",
        feature1="float32",
        feature2="bytes",
        feature3="int64",
        feature4="float64",
        feature5="string",
        feature6="int32",
        feature7="float32",
        feature8="bytes",
        feature9="int64",
    )
    generate_schema(
        yaml_path="temp/follower_schema.yaml",
        oaid="string",
        feature10="float64",
        feature11="string",
        feature12="int32",
        feature13="float32",
        feature14="bytes",
        feature15="int64",
        feature16="float64",
        feature17="string",
        feature18="int32",
        feature19="float32",
    )

    leader_process = Process(target=worker_process_fun, args=("leader", "server", "client", "127.0.0.1:6969",
                                                              "127.0.0.1:9696"))
    leader_process.start()
    follower_process = Process(target=worker_process_fun, args=("follower", "client", "server", "127.0.0.1:9696",
                                                                "127.0.0.1:6969"))
    follower_process.start()

    leader_process.join(timeout=10)
    follower_process.join(timeout=10)
    leader_process.terminate()
    follower_process.terminate()
    leader_process.kill()
    follower_process.kill()

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
