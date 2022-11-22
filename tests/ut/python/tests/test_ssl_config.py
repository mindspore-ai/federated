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
"""Test the functions of server in server mode FEDERATED_LEARNING"""

import numpy as np
from mindspore_federated import FeatureMap, SSLConfig

from common import check_feature_map
from common import fl_name_with_idx, make_yaml_config, start_fl_server, fl_test
from common import get_default_ssl_config, get_default_redis_ssl_config, start_redis_with_ssl
from common import start_fl_job_expect_success, update_model_expect_success, get_model_expect_success

start_fl_job_reach_threshold_rsp = "Current amount for startFLJob has reached the threshold"
update_model_reach_threshold_rsp = "Current amount for updateModel is enough."


def create_default_feature_map():
    update_feature_map = {"feature_conv": np.random.randn(2, 3).astype(np.float32),
                          "feature_bn": np.random.randn(1).astype(np.float32),
                          "feature_bn2": np.random.randn(1).astype(np.float32).reshape(tuple()),  # scalar
                          "feature_conv2": np.random.randn(2, 3).astype(np.float32)}
    return update_feature_map


@fl_test
def test_ssl_config_three_server_two_client_one_iterations_success():
    """
    Feature: Server
    Description: Test the function of aggregation of three server with two client.
    Expectation: The aggregation weights of all servers meets the expectation.
    """
    start_redis_with_ssl()
    _, _, client_crt, client_key, ca_crt = get_default_redis_ssl_config()
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    http_server_address2 = "127.0.0.1:3002"
    http_server_address3 = "127.0.0.1:3003"

    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    client_ssl_config = {"distributed_cache.cacert_filename": ca_crt,
                         "distributed_cache.cert_filename": client_crt,
                         "distributed_cache.private_key_filename": client_key}
    make_yaml_config(fl_name, client_ssl_config, output_yaml_file=yaml_config_file, start_fl_job_threshold=2,
                     enable_ssl=True)

    _, _, _, server_password, client_password = get_default_ssl_config()
    ssl_config = SSLConfig(server_password, client_password)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)
    # start three servers
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address,
                    ssl_config=ssl_config)
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address2,
                    ssl_config=ssl_config)
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address3,
                    ssl_config=ssl_config)

    iteration = 1
    # start fl job first for fl_id
    data_size = 16
    fl_id = "fl_id_xxxx1"
    start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size, enable_ssl=True)

    # start fl job second for fl_id2, visit server2
    data_size2 = 8
    fl_id2 = "fl_id_xxxx2"
    start_fl_job_expect_success(http_server_address2, fl_name, fl_id2, data_size2, enable_ssl=True)

    # update model, server2, fl_id1
    update_feature_map = create_default_feature_map()
    update_model_expect_success(http_server_address2, fl_name, fl_id, iteration, update_feature_map,
                                enable_ssl=True)

    # update model, server3, fl_id2
    update_feature_map2 = create_default_feature_map()
    update_model_expect_success(http_server_address3, fl_name, fl_id2, iteration, update_feature_map2,
                                enable_ssl=True)

    expect_feature_map = {}
    for key in ["feature_conv", "feature_bn", "feature_bn2"]:
        expect_feature_map[key] = (update_feature_map[key] + update_feature_map2[key]) / (data_size + data_size2)
    # get model from sever1
    client_feature_map, _ = get_model_expect_success(http_server_address, fl_name, iteration, enable_ssl=True)
    check_feature_map(expect_feature_map, client_feature_map)
    # get model from sever2
    client_feature_map, _ = get_model_expect_success(http_server_address2, fl_name, iteration, enable_ssl=True)
    check_feature_map(expect_feature_map, client_feature_map)
    # get model from sever3
    client_feature_map, _ = get_model_expect_success(http_server_address3, fl_name, iteration, enable_ssl=True)
    check_feature_map(expect_feature_map, client_feature_map)


@fl_test
def test_ssl_config_multi_server_enable_ssl_not_match_failed():
    """
    Feature: Server
    Description: hyper params of enable_ssl != value of first server
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, start_fl_job_threshold=2, enable_ssl=True)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    try:
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert "The client password's value is empty" in str(e)


@fl_test
def test_ssl_config_server_password_error_start_failed():
    """
    Feature: Server
    Description: Error server password, error client password
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, start_fl_job_threshold=2, enable_ssl=True)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    _, _, _, server_password, client_password = get_default_ssl_config()
    try:
        ssl_config = SSLConfig(server_password + "error", client_password)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address,
                        ssl_config=ssl_config)
        assert False
    except RuntimeError as e:
        assert "PKCS12_parse failed" in str(e)

    try:
        ssl_config = SSLConfig(server_password, client_password + "error")
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address,
                        ssl_config=ssl_config)
        assert False
    except RuntimeError as e:
        assert "PKCS12_parse failed" in str(e)
