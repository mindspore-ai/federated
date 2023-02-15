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

import random
import numpy as np
from mindspore import context, Model
import mindspore.nn as nn
import mindspore_federated
from mindspore_federated import FeatureMap
from mindspore_federated import FederatedLearningManager
from mindspore_federated import log as logger
from mindspore_federated.startup.ssl_config import SSLConfig

from common import create_dataset, post_scheduler_state_msg, get_model_expect_success, check_feature_map
from common import fl_name_with_idx, make_yaml_config, start_fl_server, fl_test
from common import run_worker_client_with_train_task, wait_worker_client_task_result
from common import start_fl_scheduler, stop_processes, WorkerParam
from common import start_redis_with_ssl, get_default_redis_ssl_config, read_metrics
from common import start_fl_job_expect_success, update_model_expect_success
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


def get_expect_feature_map(update_feature_map, update_feature_map2, data_size, data_size2):
    expect_feature_map = {}
    for name, _ in update_feature_map.items():
        worker0_val = update_feature_map[name] * data_size
        worker1_val = update_feature_map2[name] * data_size2
        expect_feature_map[name] = (worker0_val + worker1_val) / (data_size + data_size2)
    return expect_feature_map


def worker_fun_with_train(worker_param: WorkerParam):
    """ worker fun with train function"""
    context.set_context(mode=context.GRAPH_MODE)
    logger.info("num_batches：{}, sync_frequency: {}".format(worker_param.dataset.get_dataset_size(),
                                                            worker_param.sync_frequency))
    net_opt = nn.Momentum(worker_param.network.trainable_params(), 0.01, 0.9)
    net_loss = nn.SoftmaxCrossEntropyWithLogits()
    model = Model(worker_param.network, net_loss, net_opt)
    cbs = list()
    federated_learning_manager = FederatedLearningManager(
        yaml_config=worker_param.yaml_config_file,
        model=worker_param.network,
        sync_frequency=worker_param.sync_frequency,
        http_server_address=worker_param.http_server_address,
        data_size=worker_param.data_size,
        ssl_config=worker_param.ssl_config
    )
    cbs.append(federated_learning_manager)
    for iteration_num in range(1, worker_param.iteration + 1):
        model.train(worker_param.epoch, worker_param.dataset, callbacks=cbs,
                    dataset_sink_mode=worker_param.data_sink_mode)
        client_feature_map, get_model_rsp = get_model_expect_success(worker_param.http_server_address, "FlTest",
                                                                     iteration_num, enable_ssl=worker_param.enable_ssl)
        assert isinstance(get_model_rsp, ResponseGetModel.ResponseGetModel)
        expect_feature_map = get_trainable_params(worker_param.network)
        check_feature_map(expect_feature_map, client_feature_map)


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
    fl_iteration_num = 3
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=fl_iteration_num,
                     server_mode="CLOUD_TRAINING", start_fl_job_threshold=2, update_model_ratio=1.0,
                     start_fl_job_time_window=60000, update_model_time_window=60000, global_iteration_time_window=60000)
    server_network = LeNet5(10, 1)
    scheduler_http_address = "127.0.0.1:4000"
    worker_network = LeNet5(10, 1)
    worker_network2 = LeNet5(10, 1)
    data_size = random.randint(1000, 9999)
    data_size2 = random.randint(1000, 9999)

    dataset = create_dataset()
    num_batches = dataset.get_dataset_size()
    epoch = 2
    sync_frequency = epoch * num_batches
    worker_param = WorkerParam(yaml_config_file=yaml_config_file, network=worker_network, sync_frequency=sync_frequency,
                               http_server_address=http_server_address, data_size=data_size,
                               epoch=epoch, iteration=fl_iteration_num, dataset=dataset)

    worker_param2 = WorkerParam(yaml_config_file=yaml_config_file, network=worker_network2,
                                sync_frequency=sync_frequency,
                                http_server_address=http_server_address, data_size=data_size2,
                                epoch=epoch, iteration=fl_iteration_num, dataset=dataset)

    feature_map = FeatureMap()
    init_feature_map = get_trainable_params(server_network)
    for param_name, param_np in init_feature_map.items():
        print(f"----------------{param_name} {param_np.shape}")
        feature_map.add_feature(param_name, param_np, require_aggr=True)

    start_fl_server(feature_map, yaml_config_file, http_server_address)
    start_fl_scheduler(yaml_config_file, scheduler_http_address)
    worker_process, worker_recv_pipe = run_worker_client_with_train_task(worker_fun_with_train, worker_param)
    worker_process2, worker_recv_pipe2 = run_worker_client_with_train_task(worker_fun_with_train, worker_param2)
    wait_worker_client_task_result(worker_process, worker_recv_pipe, max_run_secs=600)
    wait_worker_client_task_result(worker_process2, worker_recv_pipe2, max_run_secs=600)

    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    print("get state:", post_rsp)
    assert "code" in post_rsp and post_rsp.get("code") == "0"
    assert "cluster_state" in post_rsp and post_rsp.get("cluster_state") == "CLUSTER_FINISH"
    assert "nodes" in post_rsp and len(post_rsp.get("nodes")) == 1
    node0 = post_rsp.get("nodes")[0]
    assert node0.get("tcp_address") in node0.get("node_id")
    assert node0.get("role") == "SERVER"

    assert stop_processes(worker_process)
    assert stop_processes(worker_process2)


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
    fl_iteration_num = 3
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=fl_iteration_num,
                     server_mode="CLOUD_TRAINING", start_fl_job_threshold=2, update_model_ratio=1.0,
                     start_fl_job_time_window=60000, update_model_time_window=60000, global_iteration_time_window=60000)
    server_network = LeNet5(10, 1)
    scheduler_http_address = "127.0.0.1:4000"
    worker_network = LeNet5(10, 1)
    worker_network2 = LeNet5(10, 1)
    data_size = random.randint(1000, 9999)
    data_size2 = random.randint(1000, 9999)

    dataset = create_dataset()
    num_batches = dataset.get_dataset_size()
    epoch = 2
    sync_frequency = epoch * num_batches
    worker_param = WorkerParam(yaml_config_file=yaml_config_file, network=worker_network, sync_frequency=sync_frequency,
                               http_server_address=http_server_address, data_size=data_size,
                               epoch=epoch, iteration=fl_iteration_num, dataset=dataset)

    worker_param2 = WorkerParam(yaml_config_file=yaml_config_file, network=worker_network2,
                                sync_frequency=sync_frequency,
                                http_server_address=http_server_address2, data_size=data_size2,
                                epoch=epoch, iteration=fl_iteration_num, dataset=dataset)

    feature_map = FeatureMap()
    init_feature_map = get_trainable_params(server_network)
    for param_name, param_np in init_feature_map.items():
        print(f"----------------{param_name} {param_np.shape}")
        feature_map.add_feature(param_name, param_np, require_aggr=True)

    start_fl_server(feature_map, yaml_config_file, http_server_address)
    start_fl_server(feature_map, yaml_config_file, http_server_address2)

    start_fl_scheduler(yaml_config_file, scheduler_http_address)
    worker_process, worker_recv_pipe = run_worker_client_with_train_task(worker_fun_with_train, worker_param)
    worker_process2, worker_recv_pipe2 = run_worker_client_with_train_task(worker_fun_with_train, worker_param2)
    wait_worker_client_task_result(worker_process, worker_recv_pipe, max_run_secs=600)
    wait_worker_client_task_result(worker_process2, worker_recv_pipe2, max_run_secs=600)

    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    print("get state:", post_rsp)
    assert "code" in post_rsp and post_rsp.get("code") == "0"
    assert "cluster_state" in post_rsp and post_rsp.get("cluster_state") == "CLUSTER_FINISH"
    assert "nodes" in post_rsp and len(post_rsp.get("nodes")) == 2
    node0 = post_rsp.get("nodes")[0]
    assert node0.get("tcp_address") in node0.get("node_id")
    assert node0.get("role") == "SERVER"

    assert stop_processes(worker_process)
    assert stop_processes(worker_process2)


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
    scheduler_http_address = "127.0.0.1:4000"
    server_network = LeNet5(10, 1)
    worker_network = LeNet5(10, 1)
    worker_network2 = LeNet5(10, 1)
    data_size = random.randint(1000, 9999)
    data_size2 = random.randint(1000, 9999)
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 3
    make_yaml_config(fl_name, client_ssl_config, output_yaml_file=yaml_config_file,
                     fl_iteration_num=fl_iteration_num,
                     server_mode="CLOUD_TRAINING", start_fl_job_threshold=2, update_model_ratio=1.0,
                     start_fl_job_time_window=60000, update_model_time_window=60000, global_iteration_time_window=60000,
                     enable_ssl=True)

    dataset = create_dataset()
    num_batches = dataset.get_dataset_size()
    epoch = 2
    sync_frequency = epoch * num_batches
    worker_param = WorkerParam(yaml_config_file=yaml_config_file, network=worker_network, sync_frequency=sync_frequency,
                               http_server_address=http_server_address, data_size=data_size, ssl_config=ssl_config,
                               epoch=epoch, iteration=fl_iteration_num, dataset=dataset, enable_ssl=True)

    worker_param2 = WorkerParam(yaml_config_file=yaml_config_file, network=worker_network2,
                                sync_frequency=sync_frequency,
                                http_server_address=http_server_address, data_size=data_size2, ssl_config=ssl_config,
                                epoch=epoch, iteration=fl_iteration_num, dataset=dataset, enable_ssl=True)

    feature_map = FeatureMap()
    init_feature_map = get_trainable_params(server_network)
    for param_name, param_np in init_feature_map.items():
        feature_map.add_feature(param_name, param_np, require_aggr=True)

    start_fl_server(feature_map, yaml_config_file, http_server_address, ssl_config=ssl_config)
    start_fl_scheduler(yaml_config_file, scheduler_http_address, ssl_config=ssl_config)

    worker_process, worker_recv_pipe = run_worker_client_with_train_task(worker_fun_with_train, worker_param)
    worker_process2, worker_recv_pipe2 = run_worker_client_with_train_task(worker_fun_with_train, worker_param2)
    wait_worker_client_task_result(worker_process, worker_recv_pipe, max_run_secs=600)
    wait_worker_client_task_result(worker_process2, worker_recv_pipe2, max_run_secs=600)

    post_rsp = post_scheduler_state_msg(scheduler_http_address, enable_ssl=True)
    print("get state:", post_rsp)
    assert "code" in post_rsp and post_rsp.get("code") == "0"
    assert "cluster_state" in post_rsp and post_rsp.get("cluster_state") == "CLUSTER_FINISH"
    assert "nodes" in post_rsp and len(post_rsp.get("nodes")) == 1
    node0 = post_rsp.get("nodes")[0]
    assert node0.get("tcp_address") in node0.get("node_id")
    assert node0.get("role") == "SERVER"

    assert stop_processes(worker_process)
    assert stop_processes(worker_process2)


def hybrid_worker_fun_with_train(worker_param: WorkerParam):
    """ worker fun with train function"""
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    update_feature_map = get_trainable_params(worker_param.network)

    net_opt = nn.Momentum(worker_param.network.trainable_params(), 0.01, 0.9)
    net_loss = nn.SoftmaxCrossEntropyWithLogits()
    model = Model(worker_param.network, net_loss, net_opt)
    cbs = list()
    federated_learning_manager = FederatedLearningManager(
        yaml_config=worker_param.yaml_config_file,
        model=worker_param.network,
        sync_frequency=worker_param.sync_frequency,
        http_server_address=worker_param.http_server_address,
        data_size=worker_param.data_size,
        ssl_config=worker_param.ssl_config
    )
    cbs.append(federated_learning_manager)
    push_metrics = mindspore_federated.PushMetrics()
    fl_id = "1xxx"
    for iteration_num in range(1, worker_param.iteration + 1):
        # start fl job
        start_fl_job_expect_success(worker_param.http_server_address, "FlTest", fl_id, 32)
        # update model
        update_model_expect_success(worker_param.http_server_address, "FlTest", fl_id, iteration_num,
                                    update_feature_map)

        model.train(worker_param.epoch, worker_param.dataset, callbacks=cbs,
                    dataset_sink_mode=worker_param.data_sink_mode)
        push_metrics.construct(0.1, 0.2)
        client_feature_map, get_model_rsp = get_model_expect_success(worker_param.http_server_address, "FlTest",
                                                                     iteration_num,
                                                                     enable_ssl=worker_param.enable_ssl)
        assert isinstance(get_model_rsp, ResponseGetModel.ResponseGetModel)

        expect_feature_map = get_trainable_params(worker_param.network)
        check_feature_map(expect_feature_map, client_feature_map)


@fl_test
def test_hybrid_one_server_success_with_worker_train():
    """
    Feature: Test FL in server mode HYBRID_TRAINING
    Description: Test the whole process of Hybrid mode with one server and one worker with train mode.
    Expectation: The server and worker work as expected.
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 5
    server_mode = "HYBRID_TRAINING"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file,
                     fl_iteration_num=fl_iteration_num,
                     server_mode=server_mode, start_fl_job_threshold=1, update_model_ratio=1.0,
                     start_fl_job_time_window=60000, update_model_time_window=60000,
                     global_iteration_time_window=60000)

    server_network = LeNet5(10, 1)
    worker_network = LeNet5(10, 1)
    dataset = create_dataset()
    num_batches = dataset.get_dataset_size()
    epoch = 2
    sync_frequency = epoch * num_batches
    feature_map = FeatureMap()
    init_feature_map = get_trainable_params(server_network)
    for param_name, param_np in init_feature_map.items():
        print(f"----------------{param_name} {param_np.shape}")
        feature_map.add_feature(param_name, param_np, require_aggr=True)

    start_fl_server(feature_map, yaml_config_file, http_server_address)
    scheduler_http_address = "127.0.0.1:4000"
    start_fl_scheduler(yaml_config_file, scheduler_http_address)

    worker_param = WorkerParam(yaml_config_file=yaml_config_file, network=worker_network,
                               sync_frequency=sync_frequency,
                               http_server_address=http_server_address, data_size=random.randint(1000, 9999),
                               epoch=epoch, iteration=fl_iteration_num, dataset=dataset)

    worker_process, worker_recv_pipe = run_worker_client_with_train_task(hybrid_worker_fun_with_train, worker_param)
    wait_worker_client_task_result(worker_process, worker_recv_pipe, max_run_secs=60)

    metrics = read_metrics()
    assert metrics
    last_metrics = metrics[-1]
    assert "metricsLoss" in last_metrics and last_metrics["metricsLoss"] == 0.1
    assert "metricsAuc" in last_metrics and last_metrics["metricsAuc"] == 0.2

    # stop(wait) worker process
    assert stop_processes(worker_process)
    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    print("get state:", post_rsp)
    assert "code" in post_rsp and post_rsp.get("code") == "0"
    assert "cluster_state" in post_rsp and post_rsp.get("cluster_state") == "CLUSTER_FINISH"
    assert "nodes" in post_rsp and len(post_rsp.get("nodes")) == 1
    node0 = post_rsp.get("nodes")[0]
    assert node0.get("tcp_address") in node0.get("node_id")
    assert node0.get("role") == "SERVER"
