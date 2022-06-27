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
"""Test the functions of server and worker in server mode HYBRID_TRAINING"""
import os
import subprocess
import time
from multiprocessing import Pipe
import numpy as np

from common import fl_name_with_idx, make_yaml_config, start_fl_server, g_redis_server_address, fl_test
from common_client import post_start_fl_job, post_get_model, post_update_model
from common_client import server_safemode_rsp, server_disabled_finished_rsp
from common_client import ResponseCode, ResponseFLJob, ResponseGetModel, ResponseUpdateModel
from common import start_fl_job_expect_success, update_model_expect_success, get_model_expect_success
from common import check_feature_map, post_scheduler_state_msg, start_fl_scheduler, stop_processes
from common import run_worker_client_task, wait_worker_client_task_result, read_metrics

from mindspore_federated import FeatureMap

from hybrid_train_network import LeNet5
from mindspore.train.callback import RunContext, _InternalCallbackParam
import mindspore as ms
from mindspore_federated import FederatedLearningManager
import mindspore_federated as ms_fl


def get_trainable_params(network):
    feature_map = {}
    for param in network.trainable_params():
        param_np = param.asnumpy()
        if param_np.dtype != np.float32:
            continue
        feature_map[param.name] = param_np
    return feature_map


def load_params_to_network(network, feature_map):
    parameter_dict = {}
    for param in network.trainable_params():
        param_np = param.asnumpy()
        if param_np.dtype != np.float32:
            continue
        value = feature_map[param.name]
        param_data = value.reshape(param_np.shape).astype(param_np.dtype)
        parameter_dict[param.name] = ms.Parameter(ms.Tensor(param_data), name=param.name)
    ms.load_param_into_net(network, parameter_dict)


def create_updated_params(feature_map_org):
    update_feature_map = {}
    for name, val in feature_map_org.items():
        update_feature_map[name] = np.random.randn(*val.shape).astype(val.dtype)
    return update_feature_map


@fl_test
def test_hybrid_one_server_success():
    """
    Feature: Test FL in server mode HYBRID_TRAINING
    Description: Test the whole process of Hybrid mode with one server and one worker.
    Expectation: The server and worker work as expected.
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 5
    server_mode = "HYBRID_TRAINING"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=fl_iteration_num,
                     server_mode=server_mode)

    network = LeNet5(62, 3)
    feature_map = FeatureMap()
    init_feature_map = get_trainable_params(network)
    for param_name, param_np in init_feature_map.items():
        print(f"----------------{param_name} {param_np.shape}")
        feature_map.add_feature(param_name, param_np, require_aggr=True)

    start_fl_server(feature_map, yaml_config_file, http_server_address)

    fl_id = "1xxx"
    data_size = 32
    # start fl job
    start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)

    iteration = 1
    update_feature_map = create_updated_params(init_feature_map)
    update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)
    expect_feature_map = {}
    for name, val in update_feature_map.items():
        expect_feature_map[name] = val / data_size

    # wait worker pull, push weight
    for i in range(6):
        client_feature_map, get_model_rsp = post_get_model(http_server_address, fl_name, iteration)
        assert client_feature_map is None
        time.sleep(0.5)
    assert isinstance(get_model_rsp, ResponseGetModel.ResponseGetModel)
    assert "The model is not ready yet for iteration" in get_model_rsp.Reason().decode()

    push_feature_map = create_updated_params(init_feature_map)
    loss = 6.9
    acc = 10.1

    scheduler_http_address = "127.0.0.1:4000"
    start_fl_scheduler(yaml_config_file, scheduler_http_address)

    def worker_fun():
        num_batches = 4
        # define fl manager
        federated_learning_manager = FederatedLearningManager(
            yaml_config=yaml_config_file,
            model=network,
            sync_frequency=2,
            data_size=num_batches
        )
        push_metrics = ms_fl.PushMetrics()
        callback_paras = _InternalCallbackParam()
        callback_paras.batch_num = 16
        run_context = RunContext(callback_paras)
        # pull weight from sever, and the weight will be equal to update model weight
        federated_learning_manager.step_end(run_context)

        pull_feature_map = get_trainable_params(network)
        check_feature_map(expect_feature_map, pull_feature_map)

        load_params_to_network(network, push_feature_map)
        # push weight to sever, and the model weight get from server will be equal to push weight
        federated_learning_manager.step_end(run_context)
        push_metrics.construct(loss, acc)

        # get state
        post_rsp = post_scheduler_state_msg(scheduler_http_address)
        print("get state:", post_rsp)
        assert "code" in post_rsp and post_rsp["code"] == "0"
        assert "cluster_state" in post_rsp and post_rsp["cluster_state"] == "CLUSTER_READY"
        assert "nodes" in post_rsp and len(post_rsp["nodes"]) == 2
        node0 = post_rsp["nodes"][0]
        assert node0["tcp_address"] in node0["node_id"]
        assert node0["role"] == "SERVER"
        node1 = post_rsp["nodes"][1]
        assert node1["role"] == "WORKER"

    worker_process, worker_recv_pipe = run_worker_client_task(worker_fun)
    wait_worker_client_task_result(worker_process, worker_recv_pipe, max_run_secs=6)

    # get model
    client_feature_map, get_model_rsp = get_model_expect_success(http_server_address, fl_name, iteration)
    check_feature_map(push_feature_map, client_feature_map)

    assert isinstance(get_model_rsp, ResponseGetModel.ResponseGetModel)
    metrics = read_metrics()
    assert len(metrics) > 0
    last_metrics = metrics[-1]
    assert "metricsLoss" in last_metrics and last_metrics["metricsLoss"] == loss
    assert "metricsAuc" in last_metrics and last_metrics["metricsAuc"] == acc

    # stop(wait) worker process
    assert stop_processes(worker_process)
    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    print("get state:", post_rsp)
    assert "code" in post_rsp and post_rsp["code"] == "0"
    assert "cluster_state" in post_rsp and post_rsp["cluster_state"] == "CLUSTER_READY"
    assert "nodes" in post_rsp and len(post_rsp["nodes"]) == 1
    node0 = post_rsp["nodes"][0]
    assert node0["tcp_address"] in node0["node_id"]
    assert node0["role"] == "SERVER"


@fl_test
def test_hybrid_two_server_success():
    """
    Feature: Test FL in server mode HYBRID_TRAINING
    Description: Test the whole process of Hybrid mode with two servers and one worker.
    Expectation: The servers and worker work as expected.
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    http_server_address2 = "127.0.0.1:3002"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 5
    server_mode = "HYBRID_TRAINING"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=fl_iteration_num,
                     server_mode=server_mode)

    network = LeNet5(62, 3)
    feature_map = FeatureMap()
    init_feature_map = get_trainable_params(network)
    for param_name, param_np in init_feature_map.items():
        print(f"----------------{param_name} {param_np.shape}")
        feature_map.add_feature(param_name, param_np, require_aggr=True)

    start_fl_server(feature_map, yaml_config_file, http_server_address)
    start_fl_server(feature_map, yaml_config_file, http_server_address2)

    fl_id = "1xxx"
    data_size = 32

    update_feature_map0 = create_updated_params(init_feature_map)
    expect_feature_map0 = {}
    for name, val in update_feature_map0.items():
        expect_feature_map0[name] = val / data_size

    update_feature_map1 = create_updated_params(init_feature_map)
    expect_feature_map1 = {}
    for name, val in update_feature_map1.items():
        expect_feature_map1[name] = val / data_size

    push_feature_map0 = create_updated_params(init_feature_map)
    push_feature_map1 = create_updated_params(init_feature_map)
    loss0 = 6.9
    acc0 = 10.1
    loss1 = 5.7
    acc1 = 8.4

    def client_fun():
        # client post message to the first server
        # start fl job
        iteration = 1
        start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
        update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map0,
                                    upload_loss=10.1)

        # get model from the second server
        client_feature_map, get_model_rsp = get_model_expect_success(http_server_address2, fl_name, iteration)
        check_feature_map(push_feature_map0, client_feature_map)

        # client post message to the second server
        # start fl job
        iteration = 2
        start_fl_job_expect_success(http_server_address2, fl_name, fl_id, data_size)
        update_model_expect_success(http_server_address2, fl_name, fl_id, iteration, update_feature_map1,
                                    upload_loss=20.1)

        # get model from the first server
        client_feature_map, get_model_rsp = get_model_expect_success(http_server_address, fl_name, iteration)
        check_feature_map(push_feature_map1, client_feature_map)

    def worker_fun():
        num_batches = 4
        # define fl manager
        federated_learning_manager = FederatedLearningManager(
            yaml_config=yaml_config_file,
            model=network,
            sync_frequency=2,
            data_size=num_batches
        )
        push_metrics = ms_fl.PushMetrics()
        callback_paras = _InternalCallbackParam()
        callback_paras.batch_num = 16
        run_context = RunContext(callback_paras)

        # for iteration 1
        # pull weight from sever, and the weight will be equal to update model weight
        federated_learning_manager.step_end(run_context)

        pull_feature_map = get_trainable_params(network)
        check_feature_map(expect_feature_map0, pull_feature_map)

        load_params_to_network(network, push_feature_map0)
        # push weight to sever, and the model weight get from server will be equal to push weight
        federated_learning_manager.step_end(run_context)
        push_metrics.construct(loss0, acc0)

        # for iteration 2
        # pull weight from sever, and the weight will be equal to update model weight
        federated_learning_manager.step_end(run_context)

        pull_feature_map = get_trainable_params(network)
        check_feature_map(expect_feature_map1, pull_feature_map)

        load_params_to_network(network, push_feature_map1)
        # push weight to sever, and the model weight get from server will be equal to push weight
        federated_learning_manager.step_end(run_context)
        push_metrics.construct(loss1, acc1)

    client_process, client_recv_pipe = run_worker_client_task(client_fun)
    worker_process, worker_recv_pipe = run_worker_client_task(worker_fun)
    wait_worker_client_task_result(client_process, client_recv_pipe, max_run_secs=10)
    wait_worker_client_task_result(worker_process, worker_recv_pipe, max_run_secs=10)

    metrics = read_metrics()
    assert len(metrics) > 1
    metrics0 = metrics[-2]
    assert "metricsLoss" in metrics0 and metrics0["metricsLoss"] == loss0
    assert "metricsAuc" in metrics0 and metrics0["metricsAuc"] == acc0

    metrics1 = metrics[-1]
    assert "metricsLoss" in metrics1 and metrics1["metricsLoss"] == loss1
    assert "metricsAuc" in metrics1 and metrics1["metricsAuc"] == acc1
