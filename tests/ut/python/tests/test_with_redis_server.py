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
import time

import numpy as np
from mindspore_federated import FeatureMap
from mindspore_federated.startup.ssl_config import SSLConfig

from common import check_feature_map, server_safemode_rsp
from common import fl_name_with_idx, make_yaml_config, start_fl_server, fl_test
from common import restart_redis_server, get_default_ssl_config
from common import start_fl_job_expect_success, update_model_expect_success, get_model_expect_success
from common import start_redis_with_ssl, get_default_redis_ssl_config, stop_redis_server, start_redis_server
from common_client import ResponseFLJob
from common_client import post_start_fl_job

start_fl_job_reach_threshold_rsp = "Current amount for startFLJob has reached the threshold"
update_model_reach_threshold_rsp = "Current amount for updateModel is enough."


def create_default_feature_map():
    update_feature_map = {"feature_conv": np.random.randn(2, 3).astype(np.float32),
                          "feature_bn": np.random.randn(1).astype(np.float32),
                          "feature_bn2": np.random.randn(1).astype(np.float32).reshape(tuple()),  # scalar
                          "feature_conv2": np.random.randn(2, 3).astype(np.float32)}
    return update_feature_map


@fl_test
def test_redis_server_one_server_restart_redis_success():
    """
    Feature: Server
    Description: Restart redis.
    Expectation: Server moves to next iteration with model of last iteration
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, start_fl_job_threshold=2)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    iteration = 1
    # start fl job for first fl_id
    data_size = 16
    fl_id = "fl_id_xxxx"
    start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
    # update model: success, first
    update_feature_map = create_default_feature_map()
    loss0 = 1.1
    update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map, upload_loss=loss0)

    # restart redis server and all the data will be lost
    restart_redis_server()
    # retry startFLJob until iteration number update to 2
    iteration = 2
    for _ in range(10):  # 0.5*10=5s
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
        if client_feature_map is not None:
            assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
            if fl_job_rsp.Iteration() == iteration:
                break
        time.sleep(0.5)
    assert client_feature_map is not None
    assert fl_job_rsp.Iteration() == iteration

    expect_feature_map = init_feature_map
    check_feature_map(expect_feature_map, client_feature_map)

    loss0 = 1.1
    update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map, upload_loss=loss0)
    # start fl job for second fl_id2
    data_size2 = 8
    fl_id2 = "fl_id_xxxx2"
    start_fl_job_expect_success(http_server_address, fl_name, fl_id2, data_size2)
    # update model: success, second
    update_feature_map2 = create_default_feature_map()
    loss1 = 7.9
    update_model_expect_success(http_server_address, fl_name, fl_id2, iteration, update_feature_map2, upload_loss=loss1)

    client_feature_map, _ = get_model_expect_success(http_server_address, fl_name, iteration)
    # expect_feature_map = {"feature_conv2": init_feature_map["feature_conv2"]}  # require_aggr = False
    expect_feature_map = {}  # require_aggr = False
    for key in ["feature_conv", "feature_bn", "feature_bn2"]:
        expect_feature_map[key] = (update_feature_map[key] + update_feature_map2[key]) / (data_size + data_size2)
    check_feature_map(expect_feature_map, client_feature_map)


@fl_test
def test_redis_server_after_server_started_redis_shutdown():
    """
    Feature: Server with redis server
    Description: After start servers, shutdown redis sever
    Expectation: Exception will be raised
    """
    _, _, _, _, _ = get_default_redis_ssl_config()

    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"

    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    # start three servers
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    stop_redis_server()
    iteration = 1
    # start fl job first for fl_id
    data_size = 16
    fl_id = "fl_id_xxxx1"
    client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
    assert client_feature_map is None
    # after sync with redis per 1s, server status updated to unavailable
    time.sleep(3)
    client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
    assert client_feature_map is None
    assert fl_job_rsp == server_safemode_rsp

    start_redis_server()
    # retry startFLJob until iteration number update to 2
    iteration = 2
    for _ in range(10):  # 0.5*10=5s
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
        if client_feature_map is not None:
            assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
            if fl_job_rsp.Iteration() == iteration:
                break
        time.sleep(0.5)
    assert client_feature_map is not None
    assert fl_job_rsp.Iteration() == iteration


@fl_test
def test_redis_server_start_with_ssl():
    """
    Feature: Server
    Description: test redis server start with ssl
    Expectation: start fl job successful
    """
    # restart redis server with ssl
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

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    # start three servers
    _, _, _, server_password, client_password = get_default_ssl_config()
    ssl_config = SSLConfig(server_password, client_password)
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
    update_model_expect_success(http_server_address2, fl_name, fl_id, iteration, update_feature_map, enable_ssl=True)

    # update model, server3, fl_id2
    update_feature_map2 = create_default_feature_map()
    update_model_expect_success(http_server_address3, fl_name, fl_id2, iteration, update_feature_map2, enable_ssl=True)

    # expect_feature_map = {"feature_conv2": init_feature_map["feature_conv2"]}  # require_aggr = False
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
def test_redis_server_start_with_ssl_restart_redis_with_ssl():
    """
    Feature: Server
    Description: Error server password, error client password
    Expectation: Exception will be raised
    """
    # restart redis server with ssl
    start_redis_with_ssl()
    _, _, client_crt, client_key, ca_crt = get_default_redis_ssl_config()

    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    http_server_address2 = "127.0.0.1:3002"

    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    client_ssl_config = {"distributed_cache.cacert_filename": ca_crt,
                         "distributed_cache.cert_filename": client_crt,
                         "distributed_cache.private_key_filename": client_key}
    make_yaml_config(fl_name, client_ssl_config, output_yaml_file=yaml_config_file, start_fl_job_threshold=2,
                     enable_ssl=True)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    ssl_config = SSLConfig(server_password="server_password_12345", client_password="client_password_12345")
    # start three servers
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address,
                    ssl_config=ssl_config)
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address2,
                    ssl_config=ssl_config)

    # restart server with redis
    start_redis_with_ssl()
    # start fl job first for fl_id
    data_size = 16
    fl_id = "fl_id_xxxx1"
    # retry startFLJob until iteration number update to 2
    iteration = 2
    for _ in range(10):  # 0.5*10=5s
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size,
                                                           enable_ssl=True)
        if client_feature_map is not None:
            assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
            if fl_job_rsp.Iteration() == iteration:
                break
        time.sleep(0.5)
    assert client_feature_map is not None
    assert fl_job_rsp.Iteration() == iteration

    # start fl job second for fl_id2, visit server2
    data_size2 = 8
    fl_id2 = "fl_id_xxxx2"
    for _ in range(10):  # 0.5*10=5s
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address2, fl_name, fl_id2, data_size2,
                                                           enable_ssl=True)
        if client_feature_map is not None:
            assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
            if fl_job_rsp.Iteration() == iteration:
                break
        time.sleep(0.5)
    assert client_feature_map is not None
    assert fl_job_rsp.Iteration() == iteration

    # update model, server2, fl_id1
    update_feature_map = create_default_feature_map()
    update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map, enable_ssl=True)

    # update model, server3, fl_id2
    update_feature_map2 = create_default_feature_map()
    update_model_expect_success(http_server_address2, fl_name, fl_id2, iteration, update_feature_map2, enable_ssl=True)

    # expect_feature_map = {"feature_conv2": init_feature_map["feature_conv2"]}  # require_aggr = False
    expect_feature_map = {}
    for key in ["feature_conv", "feature_bn", "feature_bn2"]:
        expect_feature_map[key] = (update_feature_map[key] + update_feature_map2[key]) / (data_size + data_size2)
    # get model from sever1
    client_feature_map, _ = get_model_expect_success(http_server_address, fl_name, iteration, enable_ssl=True)
    check_feature_map(expect_feature_map, client_feature_map)
    # get model from sever2
    client_feature_map, _ = get_model_expect_success(http_server_address2, fl_name, iteration, enable_ssl=True)
    check_feature_map(expect_feature_map, client_feature_map)


@fl_test
def fail_test_redis_server_server_with_ssl_redis_without_ssl():
    """
    Feature: Server
    Description: Error server password, error client password
    Expectation: Exception will be raised
    """
    _, _, client_crt, client_key, ca_crt = get_default_redis_ssl_config()

    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"

    # server with ssl, redis server without ssl
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    client_ssl_config = {"distributed_cache.cacert_filename": ca_crt,
                         "distributed_cache.cert_filename": client_crt,
                         "distributed_cache.private_key_filename": client_key}
    make_yaml_config(fl_name, client_ssl_config, output_yaml_file=yaml_config_file, start_fl_job_threshold=2,
                     enable_ssl=True)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    # start three servers
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address,
                    max_time_sec_wait=30)


@fl_test
def test_redis_server_server_without_ssl_redis_with_ssl():
    """
    Feature: Server
    Description: Error server password, error client password
    Expectation: Exception will be raised
    """
    start_redis_with_ssl()
    _, _, _, _, _ = get_default_redis_ssl_config()

    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"

    # server with ssl, redis server without ssl
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, start_fl_job_threshold=2)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    # sync with redis server failed
    try:
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert "Sync instance info with distributed cache failed" in str(e)


@fl_test
def fail_test_redis_server_start_with_ssl_restart_redis_without_ssl():
    """
    Feature: Server
    Description: Error server password, error client password
    Expectation: Exception will be raised
    """
    # restart redis server with ssl
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

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    # start three servers
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address2)
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address3)

    # restart redis server without ssl config
    restart_redis_server()
    # start fl job first for fl_id
    data_size = 16
    fl_id = "fl_id_xxxx1"
    # retry startFLJob until iteration number update to 2
    iteration = 2
    for _ in range(10):  # 0.5*10=5s
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
        if client_feature_map is not None:
            assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
            if fl_job_rsp.Iteration() == iteration:
                break
        time.sleep(0.5)
    assert client_feature_map is not None
    assert fl_job_rsp.Iteration() == iteration
