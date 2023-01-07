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
"""Test the functions of server and worker in cross silo mode CLOUD_TRAINING"""

import numpy as np
from mindspore_federated import FeatureMap
from mindspore_federated.startup.ssl_config import SSLConfig
from mindspore_federated import FederatedLearningManager
from mindspore.train.callback import RunContext, _InternalCallbackParam

from common import check_feature_map, post_scheduler_state_msg, start_fl_scheduler, stop_processes
from common import fl_name_with_idx, make_yaml_config, start_fl_server, fl_test
from common import get_model_expect_success, start_redis_with_ssl, get_default_redis_ssl_config
from common import run_cloud_worker_client_task, wait_worker_client_task_result
from common_client import ResponseGetModel
from train_network import LeNet5


def get_trainable_params(network):
    feature_map = {}
    for param in network.trainable_params():
        param_np = param.asnumpy()
        if param_np.dtype != np.float32:
            continue
        feature_map[param.name] = param_np
    return feature_map


def worker_fun(yaml_config_file, network, sync_frequency, http_server_address, data_size, ssl_config):
    """ define fl manager"""
    federated_learning_manager = FederatedLearningManager(
        yaml_config=yaml_config_file,
        model=network,
        sync_frequency=sync_frequency,
        http_server_address=http_server_address,
        data_size=data_size,
        ssl_config=ssl_config
    )

    callback_paras = _InternalCallbackParam()
    callback_paras.batch_num = 4
    callback_paras.epoch_num = 1
    callback_paras.cur_epoch_num = 1
    callback_paras.cur_step_num = 4
    run_context = RunContext(callback_paras)
    for _ in range(sync_frequency):
        federated_learning_manager.on_train_step_begin()
        federated_learning_manager.on_train_step_end(run_context)


def get_expect_feature_map(update_feature_map, update_feature_map2, data_size, data_size2):
    expect_feature_map = {}
    for name, _ in update_feature_map.items():
        worker0_val = update_feature_map[name] * data_size
        worker1_val = update_feature_map2[name] * data_size2
        expect_feature_map[name] = (worker0_val + worker1_val) / (data_size + data_size2)
    return expect_feature_map


@fl_test
def test_cross_silo_one_server_two_worker_success():
    """
    Feature: Test FL in cloud worker mode
    Description: Test the whole process of cloud worker mode with one server and two worker.
    Expectation: The server and worker work as expected.
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=5,
                     server_mode="CLOUD_TRAINING", start_fl_job_threshold=2, update_model_ratio=1.0,
                     start_fl_job_time_window=60000, update_model_time_window=60000, global_iteration_time_window=60000)
    server_network = LeNet5(62, 3)
    scheduler_http_address = "127.0.0.1:4000"
    worker_network = LeNet5(62, 3)
    worker_network2 = LeNet5(62, 3)
    data_size = 1500
    data_size2 = 1700
    sync_frequency = 4
    feature_map = FeatureMap()
    init_feature_map = get_trainable_params(server_network)
    for param_name, param_np in init_feature_map.items():
        print(f"----------------{param_name} {param_np.shape}")
        feature_map.add_feature(param_name, param_np, require_aggr=True)

    start_fl_server(feature_map, yaml_config_file, http_server_address)
    start_fl_scheduler(yaml_config_file, scheduler_http_address)
    worker_process, worker_recv_pipe = run_cloud_worker_client_task(worker_fun, yaml_config_file, worker_network,
                                                                    sync_frequency, http_server_address, data_size,
                                                                    None)
    worker_process2, worker_recv_pipe2 = run_cloud_worker_client_task(worker_fun, yaml_config_file, worker_network2,
                                                                      sync_frequency, http_server_address, data_size2,
                                                                      None)
    wait_worker_client_task_result(worker_process, worker_recv_pipe, max_run_secs=10)
    wait_worker_client_task_result(worker_process2, worker_recv_pipe2, max_run_secs=10)

    # get model
    iteration = 1
    worker_client_feature_map, get_model_rsp = get_model_expect_success(http_server_address, fl_name, iteration)
    assert isinstance(get_model_rsp, ResponseGetModel.ResponseGetModel)

    update_feature_map = get_trainable_params(worker_network)
    update_feature_map2 = get_trainable_params(worker_network2)
    expect_feature_map = get_expect_feature_map(update_feature_map, update_feature_map2, data_size, data_size2)
    check_feature_map(expect_feature_map, worker_client_feature_map)

    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    print("get state:", post_rsp)
    assert "code" in post_rsp and post_rsp.get("code") == "0"
    assert "cluster_state" in post_rsp and post_rsp.get("cluster_state") == "CLUSTER_READY"
    assert "nodes" in post_rsp and len(post_rsp.get("nodes")) == 1
    node0 = post_rsp.get("nodes")[0]
    assert node0.get("tcp_address") in node0.get("node_id")
    assert node0.get("role") == "SERVER"
    assert stop_processes(worker_process)


@fl_test
def test_cross_silo_two_server_two_worker_success():
    """
    Feature: Test FL in cloud worker mode
    Description: Test the whole process of cloud worker mode with two server and two worker.
    Expectation: The server and worker work as expected.
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    http_server_address2 = "127.0.0.1:3002"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=5,
                     server_mode="CLOUD_TRAINING", start_fl_job_threshold=2, update_model_ratio=1.0,
                     start_fl_job_time_window=60000, update_model_time_window=60000, global_iteration_time_window=60000)
    server_network = LeNet5(62, 3)
    scheduler_http_address = "127.0.0.1:4000"
    worker_network = LeNet5(62, 3)
    worker_network2 = LeNet5(62, 3)
    data_size = 1500
    data_size2 = 1700
    sync_frequency = 4
    feature_map = FeatureMap()
    init_feature_map = get_trainable_params(server_network)
    for param_name, param_np in init_feature_map.items():
        print(f"----------------{param_name} {param_np.shape}")
        feature_map.add_feature(param_name, param_np, require_aggr=True)

    start_fl_server(feature_map, yaml_config_file, http_server_address)
    start_fl_server(feature_map, yaml_config_file, http_server_address2)

    start_fl_scheduler(yaml_config_file, scheduler_http_address)
    worker_process, worker_recv_pipe = run_cloud_worker_client_task(worker_fun, yaml_config_file, worker_network,
                                                                    sync_frequency, http_server_address, data_size,
                                                                    None)
    worker_process2, worker_recv_pipe2 = run_cloud_worker_client_task(worker_fun, yaml_config_file, worker_network2,
                                                                      sync_frequency, http_server_address2, data_size2,
                                                                      None)
    wait_worker_client_task_result(worker_process, worker_recv_pipe, max_run_secs=10)
    wait_worker_client_task_result(worker_process2, worker_recv_pipe2, max_run_secs=10)

    # get model from 2 server
    iteration = 1
    worker_client_feature_map, get_model_rsp = get_model_expect_success(http_server_address, fl_name, iteration)
    assert isinstance(get_model_rsp, ResponseGetModel.ResponseGetModel)

    worker_client_feature_map2, get_model_rsp2 = get_model_expect_success(http_server_address2, fl_name, iteration)
    assert isinstance(get_model_rsp2, ResponseGetModel.ResponseGetModel)

    update_feature_map = get_trainable_params(worker_network)
    update_feature_map2 = get_trainable_params(worker_network2)
    expect_feature_map = get_expect_feature_map(update_feature_map, update_feature_map2, data_size, data_size2)
    check_feature_map(expect_feature_map, worker_client_feature_map)
    check_feature_map(expect_feature_map, worker_client_feature_map2)

    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    print("get state:", post_rsp)
    assert "code" in post_rsp and post_rsp.get("code") == "0"
    assert "cluster_state" in post_rsp and post_rsp.get("cluster_state") == "CLUSTER_READY"
    assert "nodes" in post_rsp and len(post_rsp.get("nodes")) == 2
    node0 = post_rsp.get("nodes")[0]
    assert node0.get("tcp_address") in node0.get("node_id")
    assert node0.get("role") == "SERVER"
    assert stop_processes(worker_process)


@fl_test
def test_cross_silo_one_server_two_worker_open_ssl_success():
    """
    Feature: Test FL in cloud worker mode
    Description: Test the whole process of cloud worker mode with one server and two worker with open ssl communication.
    Expectation: The server and worker work as expected.
    """
    # restart redis server with ssl
    start_redis_with_ssl()
    _, _, client_crt, client_key, ca_crt = get_default_redis_ssl_config()
    ssl_config = SSLConfig(server_password="server_password_12345", client_password="client_password_12345")

    client_ssl_config = {"distributed_cache.cacert_filename": ca_crt,
                         "distributed_cache.cert_filename": client_crt,
                         "distributed_cache.private_key_filename": client_key}
    server_network = LeNet5(62, 3)
    worker_network = LeNet5(62, 3)
    worker_network2 = LeNet5(62, 3)
    data_size = 1500
    data_size2 = 1700
    sync_frequency = 4
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    make_yaml_config(fl_name, client_ssl_config, output_yaml_file=yaml_config_file,
                     fl_iteration_num=5,
                     server_mode="CLOUD_TRAINING", start_fl_job_threshold=2, update_model_ratio=1.0,
                     start_fl_job_time_window=60000, update_model_time_window=60000, global_iteration_time_window=60000,
                     enable_ssl=True)

    feature_map = FeatureMap()
    init_feature_map = get_trainable_params(server_network)
    for param_name, param_np in init_feature_map.items():
        feature_map.add_feature(param_name, param_np, require_aggr=True)

    start_fl_server(feature_map, yaml_config_file, http_server_address, ssl_config=ssl_config)

    update_feature_map = get_trainable_params(worker_network)
    update_feature_map2 = get_trainable_params(worker_network2)
    worker_process, worker_recv_pipe = run_cloud_worker_client_task(worker_fun, yaml_config_file, worker_network,
                                                                    sync_frequency, http_server_address, data_size,
                                                                    ssl_config)
    worker_process2, worker_recv_pipe2 = run_cloud_worker_client_task(worker_fun, yaml_config_file, worker_network2,
                                                                      sync_frequency, http_server_address, data_size2,
                                                                      ssl_config)
    wait_worker_client_task_result(worker_process, worker_recv_pipe, max_run_secs=10)
    wait_worker_client_task_result(worker_process2, worker_recv_pipe2, max_run_secs=10)

    iteration = 1
    worker_client_feature_map, get_model_rsp = get_model_expect_success(http_server_address, fl_name, iteration,
                                                                        enable_ssl=True)
    assert isinstance(get_model_rsp, ResponseGetModel.ResponseGetModel)
    expect_feature_map = get_expect_feature_map(update_feature_map, update_feature_map2, data_size, data_size2)
    check_feature_map(expect_feature_map, worker_client_feature_map)
