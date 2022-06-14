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

from common import fl_name_with_idx, make_yaml_config, start_fl_server, g_redis_server_address, fl_test
from common import stop_processes
from common_client import post_start_fl_job, post_get_model, post_update_model
from common_client import server_safemode_rsp, server_disabled_finished_rsp
from common_client import ResponseCode, ResponseFLJob, ResponseGetModel, ResponseUpdateModel
from common import start_fl_job_expect_success, update_model_expect_success, get_model_expect_success
from common import check_feature_map, read_metrics

from mindspore_fl.schema import CompressType
from mindspore_federated import FeatureMap

start_fl_job_reach_threshold_rsp = "Current amount for startFLJob has reached the threshold"
update_model_reach_threshold_rsp = "Current amount for updateModel is enough."


def create_default_feature_map():
    update_feature_map = {"feature_conv": np.random.randn(2, 3).astype(np.float32),
                          "feature_bn": np.random.randn(1).astype(np.float32),
                          "feature_bn2": np.random.randn(1).astype(np.float32).reshape(tuple()),  # scalar
                          "feature_conv2": np.random.randn(2, 3).astype(np.float32)}
    return update_feature_map


@fl_test
def test_fl_server_one_server_one_client_multi_iterations_success():
    """
    Feature: Server
    Description: Test the function of one server: startup, processing requests, and finish iteration.
    Expectation: Server works as expected.
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 5
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=fl_iteration_num)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], requires_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], requires_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], requires_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], requires_aggr=False)

    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    fl_id = "1xxx"
    data_size = 32
    # start fl job
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == 1
    assert fl_job_rsp.IsSelected()
    assert fl_job_rsp.UploadCompressType() == CompressType.CompressType.NO_COMPRESS
    assert fl_job_rsp.DownloadCompressType() == CompressType.CompressType.NO_COMPRESS

    expect_feature_map = init_feature_map
    check_feature_map(expect_feature_map, client_feature_map)

    # update model
    update_feature_map = create_default_feature_map()
    iteration = 1
    update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)
    expect_feature_map = {"feature_conv": update_feature_map["feature_conv"] / data_size,
                          "feature_bn": update_feature_map["feature_bn"] / data_size,
                          "feature_bn2": update_feature_map["feature_bn2"] / data_size,
                          "feature_conv2": init_feature_map["feature_conv2"]}  # require_aggr = False
    # get model
    client_feature_map, get_model_rsp = get_model_expect_success(http_server_address, fl_name, iteration)
    check_feature_map(expect_feature_map, client_feature_map)

    # reject updateModel with old iteration 1
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id, iteration, update_feature_map)
    assert result is None
    assert "UpdateModel iteration number is invalid:1, current iteration:2" in update_model_rsp.Reason().decode()

    # --------------------------- for more iteration
    for i in range(2, fl_iteration_num + 1):
        iteration = i
        data_size = i + 8
        print(f"for iteration {iteration} data size {data_size}")
        # startFLJob
        client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
        assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
        assert fl_job_rsp.Iteration() == iteration
        # check feature map
        check_feature_map(expect_feature_map, client_feature_map)

        # update model
        update_feature_map = create_default_feature_map()
        update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)
        expect_feature_map = {"feature_conv": update_feature_map["feature_conv"] / data_size,
                              "feature_bn": update_feature_map["feature_bn"] / data_size,
                              "feature_bn2": update_feature_map["feature_bn2"] / data_size,
                              "feature_conv2": init_feature_map["feature_conv2"]}  # require_aggr = False
        # get model
        client_feature_map, get_model_rsp = get_model_expect_success(http_server_address, fl_name, iteration)
        check_feature_map(expect_feature_map, client_feature_map)
    # startFLJob when instance is finished
    client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
    assert client_feature_map is None
    assert fl_job_rsp == server_disabled_finished_rsp
    # updateModel when instance is finished
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id, iteration, update_feature_map)
    assert result is None
    assert update_model_rsp == server_disabled_finished_rsp


@fl_test
def test_fl_server_one_server_two_client_start_fl_job_invalid():
    """
    Feature: Server
    Description: Test the function of one server: handling startFLJob messages in various abnormal scenarios.
    Expectation: Server works as expected.
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, start_fl_job_threshold=2)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], requires_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], requires_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], requires_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], requires_aggr=False)

    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    fl_id = "1xxx"
    data_size = 16
    # startFLJob: data_size invalid
    client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size=0)
    assert client_feature_map is None
    assert "FL job data size is not enough" in fl_job_rsp.Reason().decode()
    # startFLJob: data_size invalid2
    client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size=0x10000)
    assert client_feature_map is None
    assert "FL job data size is too large" in fl_job_rsp.Reason().decode()

    # startFLJob: fl_name invalid
    # client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name + "_invalid", fl_id, data_size)
    # assert client_feature_map is None
    # assert "FL job data size is too large" in fl_job_rsp.Reason().decode()

    # startFLJob success, first
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
    # check feature map
    assert len(client_feature_map) == 4

    # reject startFLJob with an existing fl_id, store fl_id failed
    client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
    assert client_feature_map is None
    assert "Updating device metadata failed for fl id" in fl_job_rsp.Reason().decode()

    # startFLJob success, second
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address, fl_name, fl_id + "2", data_size)
    # check feature map
    assert len(client_feature_map) == 4

    # reject startFLJob with an existing fl_id, reach threshold
    client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
    assert client_feature_map is None
    assert start_fl_job_reach_threshold_rsp in fl_job_rsp.Reason().decode()

    # reject startFLJob with a new fl_id
    client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id + "xx", data_size)
    assert client_feature_map is None
    assert start_fl_job_reach_threshold_rsp in fl_job_rsp.Reason().decode()


@fl_test
def test_fl_server_one_server_two_client_update_model_invalid():
    """
    Feature: Server
    Description: Test the function of one server: handling updateModel messages in various abnormal scenarios.
    Expectation: Server works as expected.
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, start_fl_job_threshold=2)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], requires_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], requires_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], requires_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], requires_aggr=False)

    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    fl_id = "1xxx"
    iteration = 1
    data_size = 16
    # startFLJob
    client_feature_map, _ = start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
    # check feature map
    assert len(client_feature_map) == 4

    # update model: missing param
    invalid_feature_map = {"feature_bn": client_feature_map["feature_bn"] * 4,
                           "feature_conv2": client_feature_map["feature_conv2"] * 6}
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id, iteration, invalid_feature_map)
    assert result is None
    assert isinstance(update_model_rsp, ResponseUpdateModel.ResponseUpdateModel)
    assert "The updated weight of parameter feature_bn2 is missing" in update_model_rsp.Reason().decode()

    # update model: param data size invalid
    invalid_feature_map = {"feature_conv": np.ones([2, 2], dtype=np.float32),  # data size invalid
                           "feature_bn": client_feature_map["feature_bn"] * 6,
                           "feature_bn2": client_feature_map["feature_bn2"] * 6,
                           "feature_conv2": client_feature_map["feature_conv2"] * 6}
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id, iteration, invalid_feature_map)
    assert result is None
    assert isinstance(update_model_rsp, ResponseUpdateModel.ResponseUpdateModel)
    assert "Verify model feature map failed" in update_model_rsp.Reason().decode()
    assert update_model_rsp.Retcode() == ResponseCode.ResponseCode.RequestError

    valid_feature_map = create_default_feature_map()
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id + "_invalid", iteration,
                                                 valid_feature_map)
    assert result is None
    assert isinstance(update_model_rsp, ResponseUpdateModel.ResponseUpdateModel)
    assert "devices_meta for 1xxx_invalid is not set" in update_model_rsp.Reason().decode()
    assert update_model_rsp.Retcode() == ResponseCode.ResponseCode.OutOfTime

    # update model: invalid iteration num
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id, iteration + 1,
                                                 valid_feature_map)
    assert result is None
    assert isinstance(update_model_rsp, ResponseUpdateModel.ResponseUpdateModel)
    assert "UpdateModel iteration number is invalid:2, current iteration:1" in update_model_rsp.Reason().decode()
    assert update_model_rsp.Retcode() == ResponseCode.ResponseCode.OutOfTime

    # update model: success, first
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id, iteration,
                                                 valid_feature_map)
    assert result is not None

    # reject update model: with an existing fl_id, store fl_id failed
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id, iteration,
                                                 valid_feature_map)
    assert result is None
    assert isinstance(update_model_rsp, ResponseUpdateModel.ResponseUpdateModel)
    assert "Updating metadata of UpdateModelClientList failed for fl id 1xxx" in update_model_rsp.Reason().decode()
    assert update_model_rsp.Retcode() == ResponseCode.ResponseCode.OutOfTime

    # start fl job for second fl_id2
    data_size2 = 8
    fl_id2 = fl_id + "_2"
    client_feature_map, _ = start_fl_job_expect_success(http_server_address, fl_name, fl_id2, data_size2)
    # check feature map
    assert len(client_feature_map) == 4
    # update model: success, second
    valid_feature_map = create_default_feature_map()
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id2, iteration,
                                                 valid_feature_map)
    assert result is not None

    # update model: enough
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id, iteration,
                                                 valid_feature_map)
    assert result is None
    assert isinstance(update_model_rsp, ResponseUpdateModel.ResponseUpdateModel)
    assert update_model_reach_threshold_rsp in update_model_rsp.Reason().decode()
    assert update_model_rsp.Retcode() == ResponseCode.ResponseCode.OutOfTime


@fl_test
def test_fl_server_one_server_two_client_all_reduce_success():
    """
    Feature: Server
    Description: Test the function of aggregation of one server with two client.
    Expectation: The aggregation weights of all servers meets the expectation.
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, start_fl_job_threshold=2)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], requires_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], requires_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], requires_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], requires_aggr=False)

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
    # start fl job for second fl_id2
    data_size2 = 8
    fl_id2 = "fl_id_xxxx2"
    start_fl_job_expect_success(http_server_address, fl_name, fl_id2, data_size2)
    # update model: success, second
    update_feature_map2 = create_default_feature_map()
    loss1 = 7.9
    update_model_expect_success(http_server_address, fl_name, fl_id2, iteration, update_feature_map2, upload_loss=loss1)

    client_feature_map, get_model_rsp = get_model_expect_success(http_server_address, fl_name, iteration)
    expect_feature_map = {"feature_conv2": init_feature_map["feature_conv2"]}  # require_aggr = False
    for key in ["feature_conv", "feature_bn", "feature_bn2"]:
        expect_feature_map[key] = (update_feature_map[key] + update_feature_map2[key]) / (data_size + data_size2)
    check_feature_map(expect_feature_map, client_feature_map)
    metrics = read_metrics()
    assert len(metrics) > 0
    last_metrics = metrics[-1]
    assert "metricsLoss" in last_metrics and last_metrics["metricsLoss"] == (loss0 + loss1) / 2


@fl_test
def test_fl_server_two_server_two_client_multi_iterations_success():
    """
    Feature: Server
    Description: Test the function of aggregation of two server with two client.
    Expectation: Server works as expected.
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    http_server_address2 = "127.0.0.1:3002"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 5
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, start_fl_job_threshold=2,
                     fl_iteration_num=fl_iteration_num)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], requires_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], requires_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], requires_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], requires_aggr=False)
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    # init server2 with different feature map, but server2 feature map will be synced from server1
    feature_map2 = FeatureMap()
    init_feature_map2 = create_default_feature_map()
    feature_map2.add_feature("feature_conv", init_feature_map2["feature_conv"], requires_aggr=True)
    feature_map2.add_feature("feature_bn", init_feature_map2["feature_bn"], requires_aggr=True)
    feature_map2.add_feature("feature_bn2", init_feature_map2["feature_bn2"], requires_aggr=True)
    feature_map2.add_feature("feature_conv2", init_feature_map2["feature_conv2"], requires_aggr=False)
    start_fl_server(feature_map=feature_map2, yaml_config=yaml_config_file, http_server_address=http_server_address2)

    iteration = 1
    # start fl job first for fl_id
    data_size = 16
    fl_id = "fl_id_xxxx1"
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
    expect_feature_map = init_feature_map
    check_feature_map(expect_feature_map, client_feature_map)

    # start fl job second for fl_id2, visit server2
    data_size2 = 8
    fl_id2 = "fl_id_xxxx2"
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address2, fl_name, fl_id2, data_size2)
    # expect feature equal of server2 with server1
    expect_feature_map = init_feature_map
    check_feature_map(expect_feature_map, client_feature_map)

    # update model, server1, fl_id2
    update_feature_map = create_default_feature_map()
    update_model_expect_success(http_server_address, fl_name, fl_id2, iteration, update_feature_map)

    # update model, server2, fl_id1
    update_feature_map2 = create_default_feature_map()
    update_model_expect_success(http_server_address2, fl_name, fl_id, iteration, update_feature_map2)

    expect_feature_map = {"feature_conv2": init_feature_map["feature_conv2"]}  # require_aggr = False
    for key in ["feature_conv", "feature_bn", "feature_bn2"]:
        expect_feature_map[key] = (update_feature_map[key] + update_feature_map2[key]) / (data_size + data_size2)
    # get model from sever1
    client_feature_map, _ = get_model_expect_success(http_server_address, fl_name, iteration)
    check_feature_map(expect_feature_map, client_feature_map)
    # get model from sever2
    client_feature_map, _ = get_model_expect_success(http_server_address2, fl_name, iteration)
    check_feature_map(expect_feature_map, client_feature_map)

    # --------------------------- for more iteration
    for i in range(2, fl_iteration_num + 1):
        iteration = i
        data_size = i + 8
        data_size2 = i + 8
        print(f"for iteration {iteration} data size {data_size} {data_size2}")
        # startFLJob, server1, fl_id1
        client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
        assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
        assert fl_job_rsp.Iteration() == iteration
        # check feature map
        check_feature_map(expect_feature_map, client_feature_map)

        # startFLJob, server2, fl_id2
        client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address2, fl_name, fl_id2, data_size2)
        assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
        assert fl_job_rsp.Iteration() == iteration
        # check feature map
        check_feature_map(expect_feature_map, client_feature_map)

        # update model, server1, fl_id1
        update_feature_map = create_default_feature_map()
        update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)

        # update model, server2, fl_id2
        update_feature_map2 = create_default_feature_map()
        update_model_expect_success(http_server_address2, fl_name, fl_id2, iteration, update_feature_map2)

        expect_feature_map = {"feature_conv2": init_feature_map["feature_conv2"]}  # require_aggr = False
        for key in ["feature_conv", "feature_bn", "feature_bn2"]:
            expect_feature_map[key] = (update_feature_map[key] + update_feature_map2[key]) / (data_size + data_size2)

        # get model from sever1
        client_feature_map, _ = get_model_expect_success(http_server_address, fl_name, iteration)
        check_feature_map(expect_feature_map, client_feature_map)
        # get model from sever2
        client_feature_map, _ = get_model_expect_success(http_server_address2, fl_name, iteration)
        check_feature_map(expect_feature_map, client_feature_map)

    # server1, startFLJob when instance is finished
    client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
    assert client_feature_map is None
    assert fl_job_rsp == server_disabled_finished_rsp
    # server2, startFLJob when instance is finished
    client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address2, fl_name, fl_id2, data_size2)
    assert client_feature_map is None
    assert fl_job_rsp == server_disabled_finished_rsp

    # server1, updateModel when instance is finished
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id, iteration, update_feature_map)
    assert result is None
    assert update_model_rsp == server_disabled_finished_rsp
    # server2, updateModel when instance is finished
    result, update_model_rsp = post_update_model(http_server_address2, fl_name, fl_id2, iteration, update_feature_map)
    assert result is None
    assert update_model_rsp == server_disabled_finished_rsp


@fl_test
def test_fl_server_three_server_two_client_one_iterations_success():
    """
    Feature: Server
    Description: Test the function of aggregation of three server with two client.
    Expectation: The aggregation weights of all servers meets the expectation.
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    http_server_address2 = "127.0.0.1:3002"
    http_server_address3 = "127.0.0.1:3003"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, start_fl_job_threshold=2)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], requires_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], requires_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], requires_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], requires_aggr=False)
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    # init server2 with different feature map, but will be synced from server1
    feature_map2 = FeatureMap()
    init_feature_map2 = create_default_feature_map()
    feature_map2.add_feature("xfeature_conv", init_feature_map2["feature_conv"], requires_aggr=True)
    feature_map2.add_feature("xfeature_bn", init_feature_map2["feature_bn"], requires_aggr=True)
    feature_map2.add_feature("xfeature_bn2", init_feature_map2["feature_bn2"], requires_aggr=True)
    feature_map2.add_feature("xfeature_conv2", init_feature_map2["feature_conv2"], requires_aggr=False)
    # init server2 with different yaml config, but will be synced from server1
    yaml_config_file2 = f"temp/yaml_{fl_name}_config2.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file2, start_fl_job_threshold=1)
    start_fl_server(feature_map=feature_map2, yaml_config=yaml_config_file2, http_server_address=http_server_address2)

    # init server3 with different missing feature map, but will be synced from server1
    feature_map3 = FeatureMap()
    init_feature_map3 = create_default_feature_map()
    feature_map3.add_feature("feature_conv", init_feature_map3["feature_conv"], requires_aggr=True)
    feature_map3.add_feature("feature_conv2", init_feature_map3["feature_conv2"], requires_aggr=False)
    # init server2 with different yaml config, but will be synced from server1
    yaml_config_file3 = f"temp/yaml_{fl_name}_config3.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file3, start_fl_job_threshold=3)
    start_fl_server(feature_map=feature_map3, yaml_config=yaml_config_file3, http_server_address=http_server_address3)

    iteration = 1
    # start fl job first for fl_id
    data_size = 16
    fl_id = "fl_id_xxxx1"
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
    expect_feature_map = init_feature_map
    check_feature_map(expect_feature_map, client_feature_map)

    # start fl job second for fl_id2, visit server2
    data_size2 = 8
    fl_id2 = "fl_id_xxxx2"
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address2, fl_name, fl_id2, data_size2)
    # expect feature equal of server2 with server1
    expect_feature_map = init_feature_map
    check_feature_map(expect_feature_map, client_feature_map)

    # start fl job third, but enough, visit server3
    data_size3 = 8
    fl_id3 = "fl_id_xxxx3"
    client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address2, fl_name, fl_id3, data_size3)
    assert client_feature_map is None
    assert start_fl_job_reach_threshold_rsp in fl_job_rsp.Reason().decode()

    # update model, server2, fl_id1
    update_feature_map = create_default_feature_map()
    update_model_expect_success(http_server_address2, fl_name, fl_id, iteration, update_feature_map)

    # update model, server3, fl_id2
    update_feature_map2 = create_default_feature_map()
    update_model_expect_success(http_server_address3, fl_name, fl_id2, iteration, update_feature_map2)

    expect_feature_map = {"feature_conv2": init_feature_map["feature_conv2"]}  # require_aggr = False
    for key in ["feature_conv", "feature_bn", "feature_bn2"]:
        expect_feature_map[key] = (update_feature_map[key] + update_feature_map2[key]) / (data_size + data_size2)
    # get model from sever1
    client_feature_map, _ = get_model_expect_success(http_server_address, fl_name, iteration)
    check_feature_map(expect_feature_map, client_feature_map)
    # get model from sever2
    client_feature_map, _ = get_model_expect_success(http_server_address2, fl_name, iteration)
    check_feature_map(expect_feature_map, client_feature_map)
    # get model from sever3
    client_feature_map, _ = get_model_expect_success(http_server_address3, fl_name, iteration)
    check_feature_map(expect_feature_map, client_feature_map)


@fl_test
def test_fl_server_checkpoint_save_load_success():
    """
    Feature: Server
    Description: Test the function of save checkpoint when iteration end and load checkpoint when server restart.
    Expectation: The weights of server meets the expectation.
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 5
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=fl_iteration_num)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], requires_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], requires_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], requires_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], requires_aggr=False)

    # start for first time
    server_process = \
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    fl_id = "1xxx"
    data_size = 32
    for i in range(3):  # run for 3 iteration
        # start fl job
        start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
        # update model, when weight aggregation is done, checkpoint file will be saved in ./fl_ckpt/
        update_feature_map = create_default_feature_map()
        iteration = i + 1
        update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)
        get_model_expect_success(http_server_address, fl_name, iteration)

    # stop server, and expect terminate signal will stop server process
    assert stop_processes(server_process)

    # start for second time
    server_process = \
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    # run more 2 iteration,
    # iteration updated to 4
    iteration = 4
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == iteration
    assert fl_job_rsp.IsSelected()
    assert fl_job_rsp.UploadCompressType() == CompressType.CompressType.NO_COMPRESS
    assert fl_job_rsp.DownloadCompressType() == CompressType.CompressType.NO_COMPRESS
    # expect feature map returned from startFLJob is ok after restart server
    expect_feature_map = {"feature_conv": update_feature_map["feature_conv"] / data_size,
                          "feature_bn": update_feature_map["feature_bn"] / data_size,
                          "feature_bn2": update_feature_map["feature_bn2"] / data_size,
                          "feature_conv2": init_feature_map["feature_conv2"]}  # require_aggr = False
    check_feature_map(expect_feature_map, client_feature_map)
    # expect feature map returned from getModel is ok after restart server
    client_feature_map, _ = get_model_expect_success(http_server_address, fl_name, iteration - 1)
    check_feature_map(expect_feature_map, client_feature_map)

    # update model, when weight aggregation is done, checkpoint file will be saved in ./fl_ckpt/
    update_feature_map = create_default_feature_map()
    update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)
    get_model_expect_success(http_server_address, fl_name, iteration)

    # start fl job
    iteration = 5
    start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
    # update model, when weight aggregation is done, checkpoint file will be saved in ./fl_ckpt/
    update_feature_map = create_default_feature_map()
    update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)
    client_feature_map, get_model_rsp = get_model_expect_success(http_server_address, fl_name, iteration)
    expect_feature_map = {"feature_conv": update_feature_map["feature_conv"] / data_size,
                          "feature_bn": update_feature_map["feature_bn"] / data_size,
                          "feature_bn2": update_feature_map["feature_bn2"] / data_size,
                          "feature_conv2": init_feature_map["feature_conv2"]}  # require_aggr = False
    check_feature_map(expect_feature_map, client_feature_map)

    # stop server, and expect terminate signal will stop server process
    assert stop_processes(server_process)
    # start for third time, and instance is finished
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    # startFLJob when instance is finished
    client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
    assert client_feature_map is None
    assert fl_job_rsp == server_disabled_finished_rsp

    # get model
    client_feature_map, get_model_rsp = get_model_expect_success(http_server_address, fl_name, iteration)
    expect_feature_map = {"feature_conv": update_feature_map["feature_conv"] / data_size,
                          "feature_bn": update_feature_map["feature_bn"] / data_size,
                          "feature_bn2": update_feature_map["feature_bn2"] / data_size,
                          "feature_conv2": init_feature_map["feature_conv2"]}  # require_aggr = False
    check_feature_map(expect_feature_map, client_feature_map)


@fl_test
def test_fl_server_exit_move_next_iteration_success():
    """
    Feature: Server
    Description: When server that processed updateModel requests exited, move to next iteration
    Expectation: The server works as expected
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 5
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=fl_iteration_num,
                     start_fl_job_threshold=2)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], requires_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], requires_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], requires_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], requires_aggr=False)

    # start for first time
    server_process = \
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    fl_id = "1xxx"
    data_size = 32
    iteration = 1
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == iteration
    update_feature_map = create_default_feature_map()
    update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)

    # stop server, and expect terminate signal will stop server process
    assert stop_processes(server_process)

    # start, and iteration will move to next
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)
    iteration = 2  # iteration move to next
    for i in range(6):
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
        if client_feature_map is not None:
            break
        time.sleep(0.5)
    assert client_feature_map is not None
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == iteration
    metrics = read_metrics()
    assert len(metrics) > 0
    last_metrics = metrics[-1]
    assert "iterationResult" in last_metrics and last_metrics["iterationResult"] == "fail"


@fl_test
def test_fl_server_exit_move_next_iteration_with_two_server_success():
    """
    Feature: Server
    Description: When server that processed updateModel requests exited, move to next iteration, two servers
    Expectation: The server works as expected
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    http_server_address2 = "127.0.0.1:3002"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 5
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=fl_iteration_num,
                     start_fl_job_threshold=2)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], requires_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], requires_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], requires_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], requires_aggr=False)

    # start for first time
    server_process = \
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address2)

    fl_id = "1xxx"
    data_size = 32
    iteration = 1
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == iteration
    update_feature_map = create_default_feature_map()
    update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map)

    # stop server, and expect terminate signal will stop server process
    assert stop_processes(server_process)

    iteration = 2  # iteration move to next
    for i in range(6):
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address2, fl_name, fl_id, data_size)
        if client_feature_map is not None:
            break
        time.sleep(0.5)
    assert client_feature_map is not None
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == iteration
    metrics = read_metrics()
    assert len(metrics) > 0
    last_metrics = metrics[-1]
    assert "iterationResult" in last_metrics and last_metrics["iterationResult"] == "fail"


@fl_test
def test_fl_server_exit_no_move_next_iteration_with_two_server_success():
    """
    Feature: Server
    Description: When server that not processed updateModel requests exited, will not move to next iteration
    Expectation: The server works as expected
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    http_server_address2 = "127.0.0.1:3002"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    fl_iteration_num = 5
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, fl_iteration_num=fl_iteration_num,
                     start_fl_job_threshold=2)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], requires_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], requires_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], requires_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], requires_aggr=False)

    # start for first time
    server_process = \
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address2)

    fl_id = "1xxx"
    data_size = 32
    iteration = 1
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size)
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == iteration
    update_feature_map = create_default_feature_map()
    # update model to server2
    update_model_expect_success(http_server_address2, fl_name, fl_id, iteration, update_feature_map)

    # stop fist server, and expect terminate signal will stop server process
    assert stop_processes(server_process)

    for i in range(4):
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address2, fl_name, fl_id, data_size)
        if client_feature_map is not None:
            break
        time.sleep(0.5)
    # server will not move to next iteration
    assert client_feature_map is None
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert "Updating device metadata failed for fl id" in fl_job_rsp.Reason().decode()

    fl_id2 = "1xxx2"
    # startFLJob to server2
    client_feature_map, fl_job_rsp = start_fl_job_expect_success(http_server_address2, fl_name, fl_id2, data_size)
    assert isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob)
    assert fl_job_rsp.Iteration() == iteration
    update_feature_map = create_default_feature_map()
    # update model to server2
    update_model_expect_success(http_server_address2, fl_name, fl_id2, iteration, update_feature_map)

    # get model
    get_model_expect_success(http_server_address2, fl_name, iteration)
