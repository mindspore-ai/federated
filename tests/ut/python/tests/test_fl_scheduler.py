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

# pylint: disable=indexing-exception

"""Test the functions of scheduler in server mode FEDERATED_LEARNING"""
import time

import numpy as np
from mindspore_federated import FeatureMap

from common import check_feature_map
from common import fl_name_with_idx, make_yaml_config, fl_test
from common import post_scheduler_enable_msg, post_scheduler_disable_msg
from common import post_scheduler_new_instance_msg, post_scheduler_query_instance_msg, post_scheduler_state_msg
from common import start_fl_job_expect_success, update_model_expect_success, get_model_expect_success, \
    get_result_expect_success
from common import start_fl_server, start_fl_scheduler, stop_processes
from common_client import ResponseFLJob
from common_client import post_start_fl_job, post_get_model, post_update_model, post_get_result
from common_client import server_disabled_finished_rsp
from mindspore_fl.schema import CompressType

start_fl_job_reach_threshold_rsp = "Current amount for startFLJob has reached the threshold"
update_model_reach_threshold_rsp = "Current amount for updateModel is enough."


def create_default_feature_map():
    update_feature_map = {"feature_conv": np.random.randn(2, 3).astype(np.float32),
                          "feature_bn": np.random.randn(1).astype(np.float32),
                          "feature_bn2": np.random.randn(1).astype(np.float32).reshape(tuple()),  # scalar
                          "feature_conv2": np.random.randn(2, 3).astype(np.float32)}
    return update_feature_map


@fl_test
def test_fl_scheduler_invalid_scheduler_address():
    """
    Feature: Scheduler
    Description: Test the function of start Scheduler
    Expectation: Scheduler starts as expected
    """
    fl_name = fl_name_with_idx("FlTest")
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 5
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=fl_iteration_num)

    scheduler_http_address = "127.0.0.1:4000x"
    try:
        start_fl_scheduler(yaml_config_file, scheduler_http_address)
        assert False
    except RuntimeError as e:
        assert "Failed to start scheduler http server, invalid server address:" in str(e)

    scheduler_http_address = "127.0.0.1:"
    try:
        start_fl_scheduler(yaml_config_file, scheduler_http_address)
        assert False
    except RuntimeError as e:
        assert "Failed to start scheduler http server, invalid server address:" in str(e)

    scheduler_http_address = "127.0.0.1"
    try:
        start_fl_scheduler(yaml_config_file, scheduler_http_address)
        assert False
    except RuntimeError as e:
        assert "Failed to start scheduler http server, invalid server address:" in str(e)

    scheduler_http_address = "256.0.0.1:5000"
    try:
        start_fl_scheduler(yaml_config_file, scheduler_http_address)
        assert False
    except RuntimeError as e:
        assert "Failed to start scheduler http server, invalid server address:" in str(e)

    scheduler_http_address = 5000
    try:
        start_fl_scheduler(yaml_config_file, scheduler_http_address)
        assert False
    except RuntimeError as e:
        assert "Parameter 'manage_address' should be str" in str(e)


@fl_test
def test_fl_scheduler_post_state_success():
    """
    Feature: Scheduler
    Description: Test the function of posting /state messages to Scheduler
    Expectation: Scheduler processes the /state messages as expected
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 5
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=fl_iteration_num)

    scheduler_http_address = "127.0.0.1:4000"
    start_fl_scheduler(yaml_config_file, scheduler_http_address)
    # post scheduler message before start server
    # post /state message
    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    assert "error_message" in post_rsp and post_rsp["error_message"] == f"Cannot find cluster info for {fl_name}"
    assert "code" in post_rsp and post_rsp["code"] == "1"

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    server0 = start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                              http_server_address=http_server_address)
    # post message after start one server
    # post /state message
    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    assert "code" in post_rsp and post_rsp["code"] == "0"
    assert "cluster_state" in post_rsp and post_rsp["cluster_state"] == "CLUSTER_READY"
    assert "nodes" in post_rsp and len(post_rsp["nodes"]) == 1
    node0 = post_rsp["nodes"][0]
    assert node0["tcp_address"] in node0["node_id"]
    assert node0["role"] == "SERVER"

    # start one more server
    http_server_address2 = "127.0.0.1:3002"
    server1 = start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                              http_server_address=http_server_address2)
    # post message after start one more server
    # post /state message
    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    assert "code" in post_rsp and post_rsp["code"] == "0"
    assert "cluster_state" in post_rsp and post_rsp["cluster_state"] == "CLUSTER_READY"
    assert "nodes" in post_rsp and len(post_rsp["nodes"]) == 2
    node0 = post_rsp["nodes"][0]
    assert node0["tcp_address"] in node0["node_id"]
    assert node0["role"] == "SERVER"
    node1 = post_rsp["nodes"][1]
    assert node1["tcp_address"] in node1["node_id"]
    assert node1["role"] == "SERVER"

    # stop one server
    assert stop_processes(server0)
    # post /state message
    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    assert "code" in post_rsp and post_rsp["code"] == "0"
    assert "cluster_state" in post_rsp and post_rsp["cluster_state"] == "CLUSTER_READY"
    assert "nodes" in post_rsp and len(post_rsp["nodes"]) == 1
    # stop one more server
    assert stop_processes(server1)
    # post /state message
    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    assert "code" in post_rsp and post_rsp["code"] == "0"
    assert "nodes" not in post_rsp


@fl_test
def test_fl_scheduler_post_disable_enable_success():
    """
    Feature: Scheduler
    Description: Test the function of posting /disable and /enable messages to Scheduler
    Expectation: Scheduler processes the /disable and /enable messages as expected
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 5
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=fl_iteration_num)

    scheduler_http_address = "127.0.0.1:4000"
    start_fl_scheduler(yaml_config_file, scheduler_http_address)
    # post scheduler message before start server
    # post /disable message
    post_rsp = post_scheduler_disable_msg(scheduler_http_address)
    assert "error_message" in post_rsp and post_rsp["error_message"] == f"Cannot find cluster info for {fl_name}"
    assert "code" in post_rsp and post_rsp["code"] == "1"
    # post /enable message
    post_rsp = post_scheduler_enable_msg(scheduler_http_address)
    assert "error_message" in post_rsp and post_rsp["error_message"] == f"Cannot find cluster info for {fl_name}"
    assert "code" in post_rsp and post_rsp["code"] == "1"

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)
    # post message after start one server
    # post /disable message
    post_rsp = post_scheduler_disable_msg(scheduler_http_address)
    assert "message" in post_rsp and post_rsp["message"] == "start disabling FL-Server successful."
    assert "code" in post_rsp and post_rsp["code"] == "0"

    # repeat post /disable message
    post_rsp = post_scheduler_disable_msg(scheduler_http_address)
    assert "error_message" in post_rsp and post_rsp["error_message"] == "The instance has already been disabled."
    assert "code" in post_rsp and post_rsp["code"] == "1"

    # post /state message
    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    assert "code" in post_rsp and post_rsp["code"] == "0"
    assert "cluster_state" in post_rsp and post_rsp["cluster_state"] == "CLUSTER_DISABLE"
    assert "nodes" in post_rsp and len(post_rsp["nodes"]) == 1
    node0 = post_rsp["nodes"][0]
    assert node0["tcp_address"] in node0["node_id"]
    assert node0["role"] == "SERVER"

    fl_id = "1xxx"
    data_size = 32
    # start fl job
    for i in range(5):  # wait FL server updated to state disable
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
        if fl_job_rsp == server_disabled_finished_rsp:
            break
        time.sleep(0.5)
    assert client_feature_map is None
    assert fl_job_rsp == server_disabled_finished_rsp

    iteration = 2  # after disable FLS, iteration is update to 2
    # update model
    update_feature_map = create_default_feature_map()
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id, iteration, update_feature_map)
    assert result is None
    assert update_model_rsp == server_disabled_finished_rsp

    # get result
    result, get_result_rsp = post_get_result(http_server_address, fl_name, iteration - 1)
    assert result is None
    assert get_result_rsp == server_disabled_finished_rsp

    # get model
    result, get_model_rsp = post_get_model(http_server_address, fl_name, iteration - 1)
    assert result is None
    assert get_model_rsp == server_disabled_finished_rsp

    # post /enable message
    post_rsp = post_scheduler_enable_msg(scheduler_http_address)
    assert "message" in post_rsp and post_rsp["message"] == "start enabling FL-Server successful."
    assert "code" in post_rsp and post_rsp["code"] == "0"

    # repeat post /enable message
    post_rsp = post_scheduler_enable_msg(scheduler_http_address)
    assert "error_message" in post_rsp and post_rsp["error_message"] == "The instance has already been enabled."
    assert "code" in post_rsp and post_rsp["code"] == "1"

    # post /state message
    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    assert "code" in post_rsp and post_rsp["code"] == "0"
    assert "cluster_state" in post_rsp and post_rsp["cluster_state"] == "CLUSTER_READY"
    assert "nodes" in post_rsp and len(post_rsp["nodes"]) == 1
    node0 = post_rsp["nodes"][0]
    assert node0["tcp_address"] in node0["node_id"]
    assert node0["role"] == "SERVER"

    # start fl job
    for i in range(5):  # wait FL server updated to state enable
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
        if fl_job_rsp != server_disabled_finished_rsp:
            break
        time.sleep(0.5)
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == iteration
    assert fl_job_rsp.IsSelected()
    assert fl_job_rsp.UploadCompressType() == CompressType.CompressType.NO_COMPRESS
    assert fl_job_rsp.DownloadCompressType() == CompressType.CompressType.NO_COMPRESS

    expect_feature_map = init_feature_map
    check_feature_map(expect_feature_map, client_feature_map)

    # update model
    update_feature_map = create_default_feature_map()
    update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)
    expect_feature_map = {"feature_conv": update_feature_map["feature_conv"] / data_size,
                          "feature_bn": update_feature_map["feature_bn"] / data_size,
                          "feature_bn2": update_feature_map["feature_bn2"] / data_size}

    # get result
    get_result_expect_success(http_server_address, fl_name, iteration)

    # get model
    client_feature_map, get_model_rsp = get_model_expect_success(http_server_address, fl_name, iteration)
    check_feature_map(expect_feature_map, client_feature_map)

    for i in range(3, 6):  # for iteration 3, 4, 5
        iteration = i
        # start fl job
        start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
        update_feature_map = create_default_feature_map()
        update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)
        get_result_expect_success(http_server_address, fl_name, iteration)
        get_model_expect_success(http_server_address, fl_name, iteration)

    # post /state message, state is updated to finish
    post_rsp = post_scheduler_state_msg(scheduler_http_address)
    assert "code" in post_rsp and post_rsp["code"] == "0"
    assert "cluster_state" in post_rsp and post_rsp["cluster_state"] == "CLUSTER_FINISH"
    assert "nodes" in post_rsp and len(post_rsp["nodes"]) == 1
    node0 = post_rsp["nodes"][0]
    assert node0["tcp_address"] in node0["node_id"]
    assert node0["role"] == "SERVER"

    # post /disable message
    post_rsp = post_scheduler_disable_msg(scheduler_http_address)
    assert "error_message" in post_rsp and post_rsp[
        "error_message"] == "The instance is completed and cannot be disabled."
    assert "code" in post_rsp and post_rsp["code"] == "1"

    # post /enable message
    post_rsp = post_scheduler_enable_msg(scheduler_http_address)
    assert "error_message" in post_rsp and post_rsp[
        "error_message"] == "The instance is completed and cannot be enabled."
    assert "code" in post_rsp and post_rsp["code"] == "1"


@fl_test
def test_fl_scheduler_post_new_instance_success():
    """
    Feature: Scheduler
    Description: Test the function of posting /newInstance and /queryInstance messages to Scheduler
    Expectation: Scheduler processes the /newInstance and /queryInstance messages as expected
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 5
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=fl_iteration_num)

    scheduler_http_address = "127.0.0.1:4000"
    start_fl_scheduler(yaml_config_file, scheduler_http_address)
    # post scheduler message before start server
    # post /newInstance message
    post_rsp = post_scheduler_new_instance_msg(scheduler_http_address, {})
    assert "error_message" in post_rsp and f"Cannot find cluster info for {fl_name}" in post_rsp["error_message"]
    assert "code" in post_rsp and post_rsp["code"] == "1"
    # post /queryInstance message
    post_rsp = post_scheduler_query_instance_msg(scheduler_http_address)
    assert "error_message" in post_rsp and f"Cannot find cluster info for {fl_name}" in post_rsp["error_message"]
    assert "code" in post_rsp and post_rsp["code"] == "1"

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)
    fl_id = "1xxx"
    data_size = 32
    # start fl job, startFLJob will reach threshold
    start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)

    # post message after start one server
    # post /newInstance message
    # update start_fl_job_threshold to 2
    post_rsp = post_scheduler_new_instance_msg(scheduler_http_address, {"start_fl_job_threshold": 2})
    assert "message" in post_rsp and "Start new instance successful." in post_rsp["message"]
    # post /queryInstance message
    post_rsp = post_scheduler_query_instance_msg(scheduler_http_address)
    assert "message" in post_rsp and f"Query Instance successful." in post_rsp["message"]
    assert "code" in post_rsp and post_rsp["code"] == "0"
    assert "result" in post_rsp
    assert post_rsp["result"]["start_fl_job_threshold"] == 2

    for i in range(5):  # retry post startFLJob util newInstance handled in server
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
        if client_feature_map is not None:
            break
        time.sleep(0.5)
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == 1
    assert fl_job_rsp.IsSelected()
    assert fl_job_rsp.UploadCompressType() == CompressType.CompressType.NO_COMPRESS
    assert fl_job_rsp.DownloadCompressType() == CompressType.CompressType.NO_COMPRESS

    # update model
    update_feature_map = create_default_feature_map()
    iteration = 1
    update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)

    fl_id2 = "fl_id_2"
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address, fl_name, fl_id2, data_size)
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == 1
    update_feature_map = create_default_feature_map()
    update_model_expect_success(http_server_address, fl_name, fl_id2, iteration, update_feature_map)

    # get result
    get_result_expect_success(http_server_address, fl_name, iteration)

    # get model
    new_feature_map, _ = get_model_expect_success(http_server_address, fl_name, iteration)

    # start fl job, startFLJob will reach threshold
    start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
    start_fl_job_expect_success(http_server_address, fl_name, fl_id2, data_size)

    # post message after start one iteration
    # post /newInstance message
    # update learning rate
    post_rsp = post_scheduler_new_instance_msg(scheduler_http_address, {"client_learning_rate": 0.03})
    assert "message" in post_rsp and "Start new instance successful." in post_rsp["message"]
    # post /queryInstance message
    post_rsp = post_scheduler_query_instance_msg(scheduler_http_address)
    assert "message" in post_rsp and f"Query Instance successful." in post_rsp["message"]
    assert "code" in post_rsp and post_rsp["code"] == "0"
    assert "result" in post_rsp
    assert post_rsp["result"]["start_fl_job_threshold"] == 2
    assert abs(post_rsp["result"]["client_learning_rate"] - 0.03) < 0.001

    for i in range(5):  # retry post startFLJob util newInstance handled in server
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
        if client_feature_map is not None:
            break
        time.sleep(0.5)
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == 1
    assert abs(fl_job_rsp.FlPlanConfig().Lr() - 0.03) < 0.001
    new_feature_map["feature_conv2"] = init_feature_map["feature_conv2"]
    check_feature_map(new_feature_map, client_feature_map)

    # update model
    update_feature_map = create_default_feature_map()
    iteration = 1
    update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address, fl_name, fl_id2, data_size)
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == 1
    update_feature_map = create_default_feature_map()
    update_model_expect_success(http_server_address, fl_name, fl_id2, iteration, update_feature_map)

    # get result
    get_result_expect_success(http_server_address, fl_name, iteration)

    # get model
    get_model_expect_success(http_server_address, fl_name, iteration)

    for i in range(2, 6):  # for more than 4 iteration
        iteration = i
        # start FL job
        start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
        start_fl_job_expect_success(http_server_address, fl_name, fl_id2, data_size)
        # update model
        update_feature_map = create_default_feature_map()
        update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)
        update_feature_map = create_default_feature_map()
        update_model_expect_success(http_server_address, fl_name, fl_id2, iteration, update_feature_map)
        # get result
        get_result_expect_success(http_server_address, fl_name, iteration)
        # get model
        new_feature_map, _ = get_model_expect_success(http_server_address, fl_name, iteration)

    client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id2, data_size)
    assert client_feature_map is None
    assert fl_job_rsp == server_disabled_finished_rsp

    # post message after finish
    # update learning rate
    post_rsp = post_scheduler_new_instance_msg(scheduler_http_address, {})
    assert "message" in post_rsp and "Start new instance successful." in post_rsp["message"]
    # post /queryInstance message
    post_rsp = post_scheduler_query_instance_msg(scheduler_http_address)
    assert "message" in post_rsp and f"Query Instance successful." in post_rsp["message"]
    assert "code" in post_rsp and post_rsp["code"] == "0"
    assert "result" in post_rsp
    assert post_rsp["result"]["start_fl_job_threshold"] == 2
    assert abs(post_rsp["result"]["client_learning_rate"] - 0.03) < 0.001

    for i in range(5):  # retry post startFLJob util newInstance handled in server
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
        if client_feature_map is not None:
            break
        time.sleep(0.5)
    assert client_feature_map is not None
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == 1
    new_feature_map["feature_conv2"] = init_feature_map["feature_conv2"]
    check_feature_map(new_feature_map, client_feature_map)
