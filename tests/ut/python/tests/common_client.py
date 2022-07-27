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
"""common client"""
import requests
import numpy as np
from mindspore_fl.schema import (RequestFLJob, ResponseFLJob, ResponseCode, RequestUpdateModel, ResponseUpdateModel,
                                 FeatureMap, RequestGetModel, ResponseGetModel)
import flatbuffers


server_safemode_rsp = "The cluster is in safemode."
server_disabled_finished_rsp = "The server's training job is disabled or finished."

server_not_available_rsp = [server_safemode_rsp, server_disabled_finished_rsp]


def build_start_fl_job(fl_name, fl_id, data_size=32, timestamp="2020/11/16/19/18"):
    """build start fl job"""
    builder = flatbuffers.Builder(1024)
    fb_fl_name = builder.CreateString(fl_name)
    fb_fl_id = builder.CreateString(fl_id)
    fb_timestamp = builder.CreateString(timestamp)
    RequestFLJob.RequestFLJobStart(builder)
    RequestFLJob.RequestFLJobAddFlName(builder, fb_fl_name)
    RequestFLJob.RequestFLJobAddFlId(builder, fb_fl_id)
    RequestFLJob.RequestFLJobAddDataSize(builder, data_size)
    RequestFLJob.RequestFLJobAddTimestamp(builder, fb_timestamp)
    fl_job_request = RequestFLJob.RequestFLJobEnd(builder)
    builder.Finish(fl_job_request)
    return builder.Output()


def build_get_model(fl_name, iteration, timestamp="2020/11/16/19/18"):
    """build get model"""
    builder = flatbuffers.Builder(1024)
    fb_fl_name = builder.CreateString(fl_name)
    fb_timestamp = builder.CreateString(timestamp)
    RequestGetModel.RequestGetModelStart(builder)

    RequestGetModel.RequestGetModelAddFlName(builder, fb_fl_name)
    RequestGetModel.RequestGetModelAddTimestamp(builder, fb_timestamp)
    RequestGetModel.RequestGetModelAddIteration(builder, iteration)
    get_model_request = RequestGetModel.RequestGetModelEnd(builder)
    builder.Finish(get_model_request)
    return builder.Output()


def build_feature_map(builder, feature_map):
    """build feature map"""
    if not isinstance(builder, flatbuffers.Builder):
        raise RuntimeError("Builder should be instance of flatbuffers.Builder")
    fb_feature_list = []
    for feature_name, feature_tensor in feature_map.items():
        if not isinstance(feature_tensor, np.ndarray):
            raise RuntimeError("Feature data should be instance np.ndarray")
        FeatureMap.FeatureMapStartDataVector(builder, feature_tensor.size)
        data = feature_tensor.reshape(-1)
        for idx in range(feature_tensor.size):
            builder.PrependFloat32(data[idx])
        fb_data = builder.EndVector()
        FeatureMap.FeatureMapStart(builder)
        FeatureMap.FeatureMapAddData(builder, fb_data)
        FeatureMap.FeatureMapAddWeightFullname(builder, builder.CreateString(feature_name))
        fb_feature = FeatureMap.FeatureMapEnd(builder)
        fb_feature_list.append(fb_feature)


# feature_map: feature_name: np.ndarray
def build_update_model(fl_name, fl_id, iteration, feature_map, timestamp="2020/11/16/19/18", upload_loss=0.0):
    """build update model"""
    builder = flatbuffers.Builder(1024)
    fb_feature_list = []
    for feature_name, feature_tensor in feature_map.items():
        if not isinstance(feature_tensor, np.ndarray):
            raise RuntimeError("Feature data should be instance np.ndarray")
        fb_feature_name = builder.CreateString(feature_name)
        FeatureMap.FeatureMapStartDataVector(builder, feature_tensor.size)
        data = feature_tensor.reshape(-1)
        for idx in range(feature_tensor.size - 1, -1, -1):
            builder.PrependFloat32(data[idx])
        fb_data = builder.EndVector()
        FeatureMap.FeatureMapStart(builder)
        FeatureMap.FeatureMapAddData(builder, fb_data)
        FeatureMap.FeatureMapAddWeightFullname(builder, fb_feature_name)
        fb_feature = FeatureMap.FeatureMapEnd(builder)
        fb_feature_list.append(fb_feature)

    RequestUpdateModel.RequestUpdateModelStartFeatureMapVector(builder, len(fb_feature_list))
    for feature in fb_feature_list:
        builder.PrependUOffsetTRelative(feature)
    fb_feature_map = builder.EndVector()

    fb_fl_name = builder.CreateString(fl_name)
    fb_fl_id = builder.CreateString(fl_id)
    fb_timestamp = builder.CreateString(timestamp)
    RequestUpdateModel.RequestUpdateModelStart(builder)
    RequestUpdateModel.RequestUpdateModelAddFlName(builder, fb_fl_name)
    RequestUpdateModel.RequestUpdateModelAddFlId(builder, fb_fl_id)
    RequestUpdateModel.RequestUpdateModelAddIteration(builder, iteration)
    RequestUpdateModel.RequestUpdateModelAddFeatureMap(builder, fb_feature_map)
    RequestUpdateModel.RequestUpdateModelAddTimestamp(builder, fb_timestamp)
    RequestUpdateModel.RequestUpdateModelAddUploadLoss(builder, upload_loss)
    update_model_req = RequestUpdateModel.RequestUpdateModelEnd(builder)
    builder.Finish(update_model_req)
    return builder.Output()


class ExceptionPost:
    """ExceptionPost"""
    def __init__(self, text):
        self.text = text


def post_msg(http_address, request_url, post_data, verify=None):
    """post msg"""
    try:
        if verify:
            return requests.post(f"https://{http_address}/{request_url}", data=memoryview(post_data).tobytes(), verify=verify)
        return requests.post(f"http://{http_address}/{request_url}", data=post_data)
    except RuntimeError as e:
        return e


def post_start_fl_job(http_address, fl_name, fl_id, data_size=32, verify=None):
    """post start fl job"""
    buffer = build_start_fl_job(fl_name, fl_id, data_size)
    result = post_msg(http_address, "startFLJob", buffer, verify)
    if isinstance(result, Exception):
        raise result
    if result.text in server_not_available_rsp:
        return None, result.text

    fl_job_rsp = ResponseFLJob.ResponseFLJob.GetRootAsResponseFLJob(result.content, 0)
    if fl_job_rsp.Retcode() != ResponseCode.ResponseCode.SUCCEED:
        return None, fl_job_rsp

    feature_map = {}
    for idx in range(fl_job_rsp.FeatureMapLength()):
        feature = fl_job_rsp.FeatureMap(idx)
        if not isinstance(feature, FeatureMap.FeatureMap):
            continue
        feature_name = feature.WeightFullname().decode()
        feature_data = feature.DataAsNumpy()
        feature_map[feature_name] = feature_data
    return feature_map, fl_job_rsp


def post_update_model(http_address, fl_name, fl_id, iteration, feature_map, upload_loss=0.0, verify=None):
    """post update model"""
    buffer = build_update_model(fl_name, fl_id, iteration, feature_map, upload_loss=upload_loss)
    result = post_msg(http_address, "updateModel", buffer, verify)
    if isinstance(result, Exception):
        raise result
    if result.text in server_not_available_rsp:
        return None, result.text

    update_model_rsp = ResponseUpdateModel.ResponseUpdateModel.GetRootAsResponseUpdateModel(result.content, 0)
    if update_model_rsp.Retcode() != ResponseCode.ResponseCode.SUCCEED:
        return None, update_model_rsp

    return True, update_model_rsp


def post_get_model(http_address, fl_name, iteration, verify=None):
    """post get model"""
    buffer = build_get_model(fl_name, iteration)
    result = post_msg(http_address, "getModel", buffer, verify)
    if isinstance(result, Exception):
        raise result
    if result.text in server_not_available_rsp:
        return None, result.text

    get_model_rsp = ResponseGetModel.ResponseGetModel.GetRootAsResponseGetModel(result.content, 0)
    if get_model_rsp.Retcode() != ResponseCode.ResponseCode.SUCCEED:
        return None, get_model_rsp

    feature_map = {}
    for idx in range(get_model_rsp.FeatureMapLength()):
        feature = get_model_rsp.FeatureMap(idx)
        if not isinstance(feature, FeatureMap.FeatureMap):
            continue
        feature_name = feature.WeightFullname().decode()
        feature_data = feature.DataAsNumpy()
        feature_map[feature_name] = feature_data
    return feature_map, get_model_rsp
