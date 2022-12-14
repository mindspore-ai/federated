# Copyright 2021 Huawei Technologies Co., Ltd
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
"""client simulator"""

import argparse
import time
import datetime
import random
import sys
import requests
from mindspore.fl.schema import (RequestFLJob, ResponseFLJob, ResponseCode,
                                 RequestUpdateModel, ResponseUpdateModel,
                                 FeatureMap, RequestGetModel, ResponseGetModel)
import flatbuffers
import numpy as np

sys.path.append("../../../ut/python/tests")


parser = argparse.ArgumentParser()
parser.add_argument("--pid", type=int, default=0)
parser.add_argument("--http_type", type=str, default="http")
parser.add_argument("--http_server_address", type=str, default="127.0.0.1:6666")
parser.add_argument("--data_size", type=int, default=32)
parser.add_argument("--eval_data_size", type=int, default=32)
parser.add_argument("--upload_loss", type=float, default=1)
parser.add_argument("--upload_accuracy", type=float, default=1)

args, _ = parser.parse_known_args()
pid = args.pid
http_type = args.http_type
http_server_address = args.http_server_address
data_size = args.data_size
eval_data_size = args.eval_data_size
upload_loss = args.upload_loss
upload_accuracy = args.upload_accuracy

alphabet = 'abcdefghijklmnopqrstuvwxyz'
random_fl_id = str(random.sample(alphabet, 8))

server_not_available_rsp = ["The cluster is in safemode.",
                            "The server's training job is disabled or finished."]

def build_start_fl_job():
    """
    build start fl job
    """
    start_fl_job_builder = flatbuffers.Builder(1024)

    fl_name = start_fl_job_builder.CreateString('fl_test_job')
    fl_id = start_fl_job_builder.CreateString(random_fl_id)
    timestamp = start_fl_job_builder.CreateString('2020/11/16/19/18')

    RequestFLJob.RequestFLJobStart(start_fl_job_builder)
    RequestFLJob.RequestFLJobAddFlName(start_fl_job_builder, fl_name)
    RequestFLJob.RequestFLJobAddFlId(start_fl_job_builder, fl_id)
    RequestFLJob.RequestFLJobAddDataSize(start_fl_job_builder, data_size)
    RequestFLJob.RequestFLJobAddEvalDataSize(start_fl_job_builder, eval_data_size)
    RequestFLJob.RequestFLJobAddTimestamp(start_fl_job_builder, timestamp)
    fl_job_req = RequestFLJob.RequestFLJobEnd(start_fl_job_builder)

    start_fl_job_builder.Finish(fl_job_req)
    buf = start_fl_job_builder.Output()
    return buf


def build_feature_map(builder, feature_map_temp):
    """
    build feature map
    """
    if not feature_map_temp:
        return None
    feature_maps = []
    np_data = []
    for name, data in feature_map_temp.items():
        length = len(data)
        weight_full_name = builder.CreateString(name)
        FeatureMap.FeatureMapStartDataVector(builder, length)
        weight = np.random.rand(length) * 32
        np_data.append(weight)
        for idx in range(length - 1, -1, -1):
            builder.PrependFloat32(weight[idx])
        data = builder.EndVector()
        FeatureMap.FeatureMapStart(builder)
        FeatureMap.FeatureMapAddData(builder, data)
        FeatureMap.FeatureMapAddWeightFullname(builder, weight_full_name)
        feature_map = FeatureMap.FeatureMapEnd(builder)
        feature_maps.append(feature_map)
    return feature_maps, np_data


def build_update_model(iteration, feature_map_temp):
    """
    build updating model
    """
    builder_update_model = flatbuffers.Builder(1)
    fl_name = builder_update_model.CreateString('fl_test_job')
    fl_id = builder_update_model.CreateString(random_fl_id)
    timestamp = builder_update_model.CreateString('2020/11/16/19/18')

    feature_maps, np_data = build_feature_map(builder_update_model, feature_map_temp)

    RequestUpdateModel.RequestUpdateModelStartFeatureMapVector(builder_update_model, len(feature_map_temp))
    for single_feature_map in feature_maps:
        builder_update_model.PrependUOffsetTRelative(single_feature_map)
    feature_map = builder_update_model.EndVector()

    RequestUpdateModel.RequestUpdateModelStart(builder_update_model)
    RequestUpdateModel.RequestUpdateModelAddFlName(builder_update_model, fl_name)
    RequestUpdateModel.RequestUpdateModelAddFlId(builder_update_model, fl_id)
    RequestUpdateModel.RequestUpdateModelAddIteration(builder_update_model, iteration)
    RequestUpdateModel.RequestUpdateModelAddFeatureMap(builder_update_model, feature_map)
    RequestUpdateModel.RequestUpdateModelAddTimestamp(builder_update_model, timestamp)
    RequestUpdateModel.RequestUpdateModelAddUploadLoss(builder_update_model, upload_loss)
    RequestUpdateModel.RequestUpdateModelAddUploadAccuracy(builder_update_model, upload_accuracy)
    req_update_model = RequestUpdateModel.RequestUpdateModelEnd(builder_update_model)
    builder_update_model.Finish(req_update_model)
    buf = builder_update_model.Output()
    return buf, np_data


def build_get_model(iteration):
    """
    build getting model
    """
    builder_get_model = flatbuffers.Builder(1)
    fl_name = builder_get_model.CreateString('fl_test_job')
    timestamp = builder_get_model.CreateString('2020/12/16/19/18')

    RequestGetModel.RequestGetModelStart(builder_get_model)
    RequestGetModel.RequestGetModelAddFlName(builder_get_model, fl_name)
    RequestGetModel.RequestGetModelAddIteration(builder_get_model, iteration)
    RequestGetModel.RequestGetModelAddTimestamp(builder_get_model, timestamp)
    req_get_model = RequestGetModel.RequestGetModelEnd(builder_get_model)
    builder_get_model.Finish(req_get_model)
    buf = builder_get_model.Output()
    return buf


def datetime_to_timestamp(datetime_obj):
    local_timestamp = time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond // 1000.0
    return local_timestamp


# weight_to_idx = {
#     "conv1.weight": 0,
#     "conv2.weight": 1,
#     "fc1.weight": 2,
#     "fc2.weight": 3,
#     "fc3.weight": 4,
#     "fc1.bias": 5,
#     "fc2.bias": 6,
#     "fc3.bias": 7
# }

session = requests.Session()
current_iteration = 1
np.random.seed(0)


def start_fl_job():
    """
    start fl job
    """
    start_fl_job_result = {}
    iteration = 0
    url = http_type + "://" + http_server_address + '/startFLJob'

    x = session.post(url, data=memoryview(build_start_fl_job()).tobytes(), verify=False)
    if x.text in server_not_available_rsp:
        start_fl_job_result['reason'] = "Restart iteration."
        start_fl_job_result['next_ts'] = datetime_to_timestamp(datetime.datetime.now()) + 500
        print("Start fl job when safemode.")
        return start_fl_job_result, iteration

    rsp_fl_job = ResponseFLJob.ResponseFLJob.GetRootAsResponseFLJob(x.content, 0)
    iteration = rsp_fl_job.Iteration()
    rsp_feature_map = {}
    if rsp_fl_job.Retcode() != ResponseCode.ResponseCode.SUCCEED:
        if rsp_fl_job.Retcode() == ResponseCode.ResponseCode.OutOfTime:
            start_fl_job_result['reason'] = "Restart iteration."
            start_fl_job_result['next_ts'] = int(rsp_fl_job.NextReqTime().decode('utf-8'))
        else:
            print("Start fl job failed, return code is ", rsp_fl_job.Retcode())
            sys.exit()
    else:
        start_fl_job_result['reason'] = "Success"
        start_fl_job_result['next_ts'] = 0
        for i in range(0, rsp_fl_job.FeatureMapLength()):
            rsp_feature_map[rsp_fl_job.FeatureMap(i).WeightFullname()] = rsp_fl_job.FeatureMap(i).DataAsNumpy()
            print("start fl job rsp weight name:{}".format(rsp_fl_job.FeatureMap(i).WeightFullname()))
            print("start fl job rsp weight size:{}".format(len(rsp_fl_job.FeatureMap(i).DataAsNumpy())))
    return start_fl_job_result, iteration, rsp_feature_map


def update_model(iteration, feature_map_temp):
    """
    update model
    """
    update_model_result = {}

    url = http_type + "://" + http_server_address + '/updateModel'
    print("Update model url:", url, ", iteration:", iteration)
    update_model_buf, update_model_np_data = build_update_model(iteration, feature_map_temp)
    x = session.post(url, data=memoryview(update_model_buf).tobytes(), verify=False)
    if x.text in server_not_available_rsp:
        update_model_result['reason'] = "Restart iteration."
        update_model_result['next_ts'] = datetime_to_timestamp(datetime.datetime.now()) + 500
        print("Update model when safemode.")
        return update_model_result, update_model_np_data

    rsp_update_model = ResponseUpdateModel.ResponseUpdateModel.GetRootAsResponseUpdateModel(x.content, 0)
    if rsp_update_model.Retcode() != ResponseCode.ResponseCode.SUCCEED:
        if rsp_update_model.Retcode() == ResponseCode.ResponseCode.OutOfTime:
            update_model_result['reason'] = "Restart iteration."
            update_model_result['next_ts'] = int(rsp_update_model.NextReqTime().decode('utf-8'))
            print("Update model out of time. Next request at ",
                  update_model_result['next_ts'], "reason:", rsp_update_model.Reason())
        else:
            print("Update model failed, return code is ", rsp_update_model.Retcode())
            sys.exit()
    else:
        update_model_result['reason'] = "Success"
        update_model_result['next_ts'] = 0
    return update_model_result, update_model_np_data


def get_model(iteration):
    """
    get model
    """
    get_model_result = {}

    url = http_type + "://" + http_server_address  + '/getModel'
    print("Get model url:", url, ", iteration:", iteration)

    while True:
        x = session.post(url, data=memoryview(build_get_model(iteration)).tobytes(), verify=False)
        if x.text in server_not_available_rsp:
            print("Get model when safemode.")
            time.sleep(0.5)
            continue

        rsp_get_model = ResponseGetModel.ResponseGetModel.GetRootAsResponseGetModel(x.content, 0)
        ret_code = rsp_get_model.Retcode()
        if ret_code == ResponseCode.ResponseCode.SUCCEED:
            break
        elif ret_code == ResponseCode.ResponseCode.SucNotReady:
            time.sleep(0.5)
            continue
        else:
            print("Get model failed, return code is ", rsp_get_model.Retcode())
            sys.exit()

    for i in range(0, rsp_get_model.FeatureMapLength()):
        print(rsp_get_model.FeatureMap(i).WeightFullname())
        sys.stdout.flush()
    print("")
    get_model_result['reason'] = "Success"
    get_model_result['next_ts'] = 0
    return get_model_result


while True:
    result, current_iteration, rsp_feature_maps = start_fl_job()
    sys.stdout.flush()
    if result['reason'] == "Restart iteration.":
        current_ts = datetime_to_timestamp(datetime.datetime.now())
        duration = result['next_ts'] - current_ts
        if duration >= 0:
            time.sleep(duration / 1000)
        continue

    result, update_data = update_model(current_iteration, rsp_feature_maps)
    sys.stdout.flush()
    if result['reason'] == "Restart iteration.":
        current_ts = datetime_to_timestamp(datetime.datetime.now())
        duration = result['next_ts'] - current_ts
        if duration >= 0:
            time.sleep(duration / 1000)
        continue

    result = get_model(current_iteration)
    sys.stdout.flush()
    if result['reason'] == "Restart iteration.":
        current_ts = datetime_to_timestamp(datetime.datetime.now())
        duration = result['next_ts'] - current_ts
        if duration >= 0:
            time.sleep(duration / 1000)
        continue

    print("")
    sys.stdout.flush()
