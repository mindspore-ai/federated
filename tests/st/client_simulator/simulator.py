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
import os
import requests
import flatbuffers
import numpy as np
from mindspore import load_checkpoint

mindspore_fl_path = os.path.abspath(os.path.join(os.getcwd(), "../../ut/python/tests/"))
sys.path.append(mindspore_fl_path)

from mindspore_fl.schema import (RequestFLJob, ResponseFLJob, ResponseCode, RequestUpdateModel, ResponseUpdateModel,
                                 RequestGetResult, ResponseGetResult, FeatureMap, RequestGetModel, ResponseGetModel,
                                 CompressFeatureMap, UnsupervisedEvalItems, UnsupervisedEvalItem)

parser = argparse.ArgumentParser()
parser.add_argument("--pid", type=int, default=0)
parser.add_argument("--http_type", type=str, default="http")
parser.add_argument("--http_server_address", type=str, default="127.0.0.1:6666")
parser.add_argument("--data_size", type=int, default=32)
parser.add_argument("--eval_data_size", type=int, default=32)
parser.add_argument("--upload_loss", type=float, default=1)
parser.add_argument("--upload_accuracy", type=float, default=1)
parser.add_argument("--checkpoint_path", type=str, default="")

args, _ = parser.parse_known_args()
pid = args.pid
http_type = args.http_type
http_server_address = args.http_server_address
data_size = args.data_size
eval_data_size = args.eval_data_size
upload_loss = args.upload_loss
upload_accuracy = args.upload_accuracy
checkpoint_path = args.checkpoint_path

alphabet = 'abcdefghijklmnopqrstuvwxyz'
random_fl_id = str(random.sample(alphabet, 8))

server_not_available_rsp = ["The cluster is in safemode.",
                            "The server's training job is disabled or finished."]

compress_type_str_map = {0: "NO_COMPRESS", 1: "DIFF_SPARSE_QUANT", 2: "QUANT"}

param_dict = {}
if os.path.exists(checkpoint_path):
    param_dict = load_checkpoint(checkpoint_path)
print(param_dict)


def build_start_fl_job():
    """
    build start fl job
    """
    start_fl_job_builder = flatbuffers.Builder(1024)

    fl_name = start_fl_job_builder.CreateString('fl_test_job')
    fl_id = start_fl_job_builder.CreateString(random_fl_id)
    timestamp = start_fl_job_builder.CreateString('2020/11/16/19/18')

    download_compress_types = [0, 2]
    RequestFLJob.RequestFLJobStartDownloadCompressTypesVector(start_fl_job_builder, len(download_compress_types))
    for download_compress_type in reversed(download_compress_types):
        start_fl_job_builder.PrependInt8(download_compress_type)
    download_compress_types_off = start_fl_job_builder.EndVector()

    RequestFLJob.RequestFLJobStart(start_fl_job_builder)
    RequestFLJob.RequestFLJobAddFlName(start_fl_job_builder, fl_name)
    RequestFLJob.RequestFLJobAddFlId(start_fl_job_builder, fl_id)
    RequestFLJob.RequestFLJobAddDataSize(start_fl_job_builder, data_size)
    RequestFLJob.RequestFLJobAddEvalDataSize(start_fl_job_builder, eval_data_size)
    RequestFLJob.RequestFLJobAddTimestamp(start_fl_job_builder, timestamp)
    RequestFLJob.RequestFLJobAddDownloadCompressTypes(start_fl_job_builder, download_compress_types_off)
    fl_job_req = RequestFLJob.RequestFLJobEnd(start_fl_job_builder)

    start_fl_job_builder.Finish(fl_job_req)
    buf = start_fl_job_builder.Output()
    return buf


def quant_compress(data_list, quant_bits):
    """quant compression"""
    min_val = min(data_list)
    max_val = max(data_list)

    quant_p1 = ((1 << quant_bits) - 1) / (max_val - min_val + 1e-10)
    quant_p2 = int(round(quant_p1 * min_val))

    compress_data_list = []
    for data in data_list:
        compress_data = round(quant_p1 * data) - quant_p2
        compress_data_list.append(compress_data)
    return min_val, max_val, compress_data_list


def construct_mask_array(param_num, upload_sparse_rate):
    retain_num = int(param_num * upload_sparse_rate)
    return [1 for _ in range(retain_num)] + [0 for _ in range(param_num - retain_num)]


def build_compress_feature_map(builder, feature_map_temp, upload_sparse_rate):
    """
    build compress feature map
    """
    if not feature_map_temp:
        return None
    compress_feature_maps = list()
    name_vec = list()
    param_num = 0
    for _, compress_feature_map in feature_map_temp.items():
        param_num += len(compress_feature_map)
    mask_array = construct_mask_array(param_num, upload_sparse_rate)

    index = 0
    for name, value in feature_map_temp.items():
        length = len(value)
        if name in param_dict:
            param_value = param_dict[name]
        else:
            param_value = np.random.rand(length) * 32
        weight_full_name = builder.CreateString(name)
        CompressFeatureMap.CompressFeatureMapStartCompressDataVector(builder, length)
        min_val, max_val, compress_data_list = quant_compress(param_value, 8)
        compress_weight = np.array(compress_data_list, dtype=np.int8)
        for idx in range(length - 1, -1, -1):
            if mask_array[index] == 1:
                builder.PrependInt8(compress_weight[idx])
            index += 1
        data = builder.EndVector()
        CompressFeatureMap.CompressFeatureMapStart(builder)
        CompressFeatureMap.CompressFeatureMapAddCompressData(builder, data)
        CompressFeatureMap.CompressFeatureMapAddWeightFullname(builder, weight_full_name)
        CompressFeatureMap.CompressFeatureMapAddMaxVal(builder, max_val)
        CompressFeatureMap.CompressFeatureMapAddMinVal(builder, min_val)
        compress_feature_map = CompressFeatureMap.CompressFeatureMapEnd(builder)
        compress_feature_maps.append(compress_feature_map)
        feature_name = builder.CreateString(name)
        name_vec.append(feature_name)
    return compress_feature_maps, name_vec


def build_unsupervised_eval_items(builder):
    """
    build unsupervised eval items
    """
    unsupervised_eval_data_size = 160
    eval_items_size = 10
    unsupervised_eval_items = []
    for i in range(0, eval_items_size):
        eval_name = builder.CreateString("eval_name_" + str(i))
        UnsupervisedEvalItem.UnsupervisedEvalItemStartEvalDataVector(builder, unsupervised_eval_data_size)
        eval_data = np.random.random(unsupervised_eval_data_size)
        for j in range(0, unsupervised_eval_data_size):
            builder.PrependFloat32(eval_data[j])
        fbs_eval_data = builder.EndVector()
        UnsupervisedEvalItem.UnsupervisedEvalItemStart(builder)
        UnsupervisedEvalItem.UnsupervisedEvalItemAddEvalData(builder, fbs_eval_data)
        UnsupervisedEvalItem.UnsupervisedEvalItemAddEvalName(builder, eval_name)
        fbs_unsupervised_eval_item = UnsupervisedEvalItem.UnsupervisedEvalItemEnd(builder)
        unsupervised_eval_items.append(fbs_unsupervised_eval_item)

    UnsupervisedEvalItems.UnsupervisedEvalItemsStartEvalItemsVector(builder, eval_items_size)
    for unsupervised_eval_item in unsupervised_eval_items:
        builder.PrependUOffsetTRelative(unsupervised_eval_item)
    fbs_unsupervised_eval_items = builder.EndVector()

    UnsupervisedEvalItems.UnsupervisedEvalItemsStart(builder)
    UnsupervisedEvalItems.UnsupervisedEvalItemsAddEvalItems(builder, fbs_unsupervised_eval_items)
    unsupervised_end = UnsupervisedEvalItems.UnsupervisedEvalItemsEnd(builder)
    return unsupervised_end


def build_feature_map(builder, feature_map_temp):
    """
    build feature map
    """
    if not feature_map_temp:
        return None
    feature_maps = []
    for name, value in feature_map_temp.items():
        length = len(value)
        if name in param_dict:
            param_value = param_dict[name]
        else:
            param_value = np.random.rand(length) * 32
        weight_full_name = builder.CreateString(name)
        FeatureMap.FeatureMapStartDataVector(builder, length)
        for idx in range(length - 1, -1, -1):
            builder.PrependFloat32(param_value[idx])
        data = builder.EndVector()
        FeatureMap.FeatureMapStart(builder)
        FeatureMap.FeatureMapAddData(builder, data)
        FeatureMap.FeatureMapAddWeightFullname(builder, weight_full_name)
        fbs_feature_map = FeatureMap.FeatureMapEnd(builder)
        feature_maps.append(fbs_feature_map)
    return feature_maps


def build_compress_update_model(iteration, feature_map_temp, upload_compress_type, upload_sparse_rate):
    """
    build updating model
    """
    builder_update_model = flatbuffers.Builder(1)
    fl_name = builder_update_model.CreateString('fl_test_job')
    fl_id = builder_update_model.CreateString(random_fl_id)
    timestamp = builder_update_model.CreateString('2020/11/16/19/18')
    compress_feature_maps, name_vec = build_compress_feature_map(builder_update_model, feature_map_temp,
                                                                 upload_sparse_rate)
    RequestUpdateModel.RequestUpdateModelStartCompressFeatureMapVector(builder_update_model, len(compress_feature_maps))
    for single_feature_map in compress_feature_maps:
        builder_update_model.PrependUOffsetTRelative(single_feature_map)
    fbs_compress_feature_maps = builder_update_model.EndVector()

    RequestUpdateModel.RequestUpdateModelStartNameVecVector(builder_update_model, len(name_vec))
    for name in reversed(name_vec):
        builder_update_model.PrependUOffsetTRelative(name)
    fbs_name_vec = builder_update_model.EndVector()

    unsupervised_pos = build_unsupervised_eval_items(builder_update_model)
    RequestUpdateModel.RequestUpdateModelStart(builder_update_model)
    RequestUpdateModel.RequestUpdateModelAddFlName(builder_update_model, fl_name)
    RequestUpdateModel.RequestUpdateModelAddFlId(builder_update_model, fl_id)
    RequestUpdateModel.RequestUpdateModelAddIteration(builder_update_model, iteration)
    RequestUpdateModel.RequestUpdateModelAddCompressFeatureMap(builder_update_model, fbs_compress_feature_maps)
    RequestUpdateModel.RequestUpdateModelAddTimestamp(builder_update_model, timestamp)
    RequestUpdateModel.RequestUpdateModelAddUploadLoss(builder_update_model, upload_loss)
    RequestUpdateModel.RequestUpdateModelAddUploadAccuracy(builder_update_model, upload_accuracy)
    RequestUpdateModel.RequestUpdateModelAddUnsupervisedEvalItems(builder_update_model, unsupervised_pos)

    RequestUpdateModel.RequestUpdateModelAddNameVec(builder_update_model, fbs_name_vec)
    RequestUpdateModel.RequestUpdateModelAddUploadCompressType(builder_update_model, upload_compress_type)
    RequestUpdateModel.RequestUpdateModelAddUploadSparseRate(builder_update_model, upload_sparse_rate)
    req_update_model = RequestUpdateModel.RequestUpdateModelEnd(builder_update_model)
    builder_update_model.Finish(req_update_model)
    buf = builder_update_model.Output()
    return buf


def build_update_model(iteration, feature_map_temp):
    """
    build updating model
    """
    builder_update_model = flatbuffers.Builder(1)
    fl_name = builder_update_model.CreateString('fl_test_job')
    fl_id = builder_update_model.CreateString(random_fl_id)
    timestamp = builder_update_model.CreateString('2020/11/16/19/18')
    feature_maps = build_feature_map(builder_update_model, feature_map_temp)
    RequestUpdateModel.RequestUpdateModelStartFeatureMapVector(builder_update_model, len(feature_map_temp))
    for single_feature_map in feature_maps:
        builder_update_model.PrependUOffsetTRelative(single_feature_map)
    fbs_feature_map = builder_update_model.EndVector()

    unsupervised_pos = build_unsupervised_eval_items(builder_update_model)
    RequestUpdateModel.RequestUpdateModelStart(builder_update_model)
    RequestUpdateModel.RequestUpdateModelAddFlName(builder_update_model, fl_name)
    RequestUpdateModel.RequestUpdateModelAddFlId(builder_update_model, fl_id)
    RequestUpdateModel.RequestUpdateModelAddIteration(builder_update_model, iteration)
    RequestUpdateModel.RequestUpdateModelAddFeatureMap(builder_update_model, fbs_feature_map)
    RequestUpdateModel.RequestUpdateModelAddTimestamp(builder_update_model, timestamp)
    RequestUpdateModel.RequestUpdateModelAddUploadLoss(builder_update_model, upload_loss)
    RequestUpdateModel.RequestUpdateModelAddUploadAccuracy(builder_update_model, upload_accuracy)
    RequestUpdateModel.RequestUpdateModelAddUnsupervisedEvalItems(builder_update_model, unsupervised_pos)
    req_update_model = RequestUpdateModel.RequestUpdateModelEnd(builder_update_model)
    builder_update_model.Finish(req_update_model)
    buf = builder_update_model.Output()
    return buf


def build_get_result(iteration):
    """build get result"""
    builder_get_result = flatbuffers.Builder(1)
    fl_name = builder_get_result.CreateString('fl_test_job')
    timestamp = builder_get_result.CreateString('2020/12/16/19/18')
    RequestGetResult.RequestGetResultStart(builder_get_result)

    RequestGetResult.RequestGetResultAddFlName(builder_get_result, fl_name)
    RequestGetResult.RequestGetResultAddTimestamp(builder_get_result, timestamp)
    RequestGetResult.RequestGetResultAddIteration(builder_get_result, iteration)
    get_result_request = RequestGetResult.RequestGetResultEnd(builder_get_result)
    builder_get_result.Finish(get_result_request)
    buf = builder_get_result.Output()
    return buf


def build_get_model(iteration):
    """
    build getting model
    """
    builder_get_model = flatbuffers.Builder(1)
    fl_name = builder_get_model.CreateString('fl_test_job')
    timestamp = builder_get_model.CreateString('2020/12/16/19/18')

    download_compress_types = [0, 2]
    RequestGetModel.RequestGetModelStartDownloadCompressTypesVector(builder_get_model, len(download_compress_types))
    for download_compress_type in reversed(download_compress_types):
        builder_get_model.PrependInt8(download_compress_type)
    fbs_download_compress_types = builder_get_model.EndVector()

    RequestGetModel.RequestGetModelStart(builder_get_model)
    RequestGetModel.RequestGetModelAddFlName(builder_get_model, fl_name)
    RequestGetModel.RequestGetModelAddIteration(builder_get_model, iteration)
    RequestGetModel.RequestGetModelAddTimestamp(builder_get_model, timestamp)
    RequestGetModel.RequestGetModelAddDownloadCompressTypes(builder_get_model, fbs_download_compress_types)
    req_get_model = RequestGetModel.RequestGetModelEnd(builder_get_model)
    builder_get_model.Finish(req_get_model)
    buf = builder_get_model.Output()
    return buf


def datetime_to_timestamp(datetime_obj):
    local_timestamp = time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond // 1000.0
    return local_timestamp


session = requests.Session()
current_iteration = 1


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

    print("Start fl job response size: ", len(x.content))
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
        download_compress_type = rsp_fl_job.DownloadCompressType()
        print("start_fl_job:download_compress_type is ", compress_type_str_map[download_compress_type])

        if download_compress_type == 0:
            for i in range(0, rsp_fl_job.FeatureMapLength()):
                rsp_feature_map[rsp_fl_job.FeatureMap(i).WeightFullname()] = rsp_fl_job.FeatureMap(i).DataAsNumpy()
        elif download_compress_type == 2:
            for i in range(0, rsp_fl_job.CompressFeatureMapLength()):
                compress_feature_map = rsp_fl_job.CompressFeatureMap(i)
                if compress_feature_map is not None:
                    rsp_feature_map[compress_feature_map.WeightFullname()] = compress_feature_map.CompressDataAsNumpy()
        else:
            print("download_compress_type: {} is invalid.".format(download_compress_type))
            sys.exit()
    return start_fl_job_result, iteration, rsp_feature_map, rsp_fl_job.UploadCompressType(), \
           rsp_fl_job.UploadSparseRate()


def update_model(iteration, feature_map_temp, upload_compress_type, upload_sparse_rate):
    """
    update model
    """
    update_model_result = {}
    print("update_model:upload_compress_type is ", compress_type_str_map[upload_compress_type])

    url = http_type + "://" + http_server_address + '/updateModel'
    if upload_compress_type == 0:
        update_model_buf = build_update_model(iteration, feature_map_temp)
    elif upload_compress_type == 1:
        update_model_buf = build_compress_update_model(iteration, feature_map_temp, upload_compress_type,
                                                       upload_sparse_rate)
    else:
        print("upload_compress_type: {} is invalid.".format(upload_compress_type))
        sys.exit()
    print("Update model url:", url, ", iteration:", iteration, "update_model_buf size:", len(update_model_buf))
    x = session.post(url, data=memoryview(update_model_buf).tobytes(), verify=False)
    if x.text in server_not_available_rsp:
        update_model_result['reason'] = "Restart iteration."
        update_model_result['next_ts'] = datetime_to_timestamp(datetime.datetime.now()) + 500
        print("Update model when safemode.")
        return update_model_result

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
    return update_model_result


def get_result(iteration):
    """
    get result
    """
    get_result_result = {}

    url = http_type + "://" + http_server_address + '/getResult'
    print("Get result url:", url, ", iteration:", iteration)
    while True:
        get_result_buf = build_get_result(iteration)
        x = session.post(url, data=memoryview(get_result_buf).tobytes(), verify=False)
        if x.text in server_not_available_rsp:
            print("Get result when safemode.")
            time.sleep(0.5)
            continue
        print("Get result response size:", len(x.content))
        rsp_get_result = ResponseGetResult.ResponseGetResult.GetRootAsResponseGetResult(x.content, 0)
        ret_code = rsp_get_result.Retcode()
        if ret_code == ResponseCode.ResponseCode.SUCCEED:
            break
        elif ret_code == ResponseCode.ResponseCode.SucNotReady:
            time.sleep(0.5)
            continue
        else:
            print("Get result failed, return code is ", rsp_get_result.Retcode())
            sys.exit()
    print("")
    get_result_result['reason'] = "Success"
    get_result_result['next_ts'] = 0
    return get_result_result


def get_model(iteration):
    """
    get model
    """
    get_model_result = {}

    url = http_type + "://" + http_server_address + '/getModel'
    print("Get model url:", url, ", iteration:", iteration)
    while True:
        get_model_buf = build_get_model(iteration)
        x = session.post(url, data=memoryview(get_model_buf).tobytes(), verify=False)
        if x.text in server_not_available_rsp:
            print("Get model when safemode.")
            time.sleep(0.5)
            continue
        print("Get model response size:", len(x.content))
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
    download_compress_type = rsp_get_model.DownloadCompressType()
    print("get_model:download_compress_type is ", compress_type_str_map[download_compress_type])
    if download_compress_type == 0:
        for i in range(0, rsp_get_model.FeatureMapLength()):
            print("get model rsp weight name:{}".format(rsp_get_model.FeatureMap(i).WeightFullname()))
    elif download_compress_type == 2:
        for i in range(0, rsp_get_model.CompressFeatureMapLength()):
            print("get model compress rsp weight name:{}".format(rsp_get_model.CompressFeatureMap(i).WeightFullname()))
    else:
        print("download_compress_type: {} is invalid.".format(download_compress_type))
        sys.exit()
    print("")
    get_model_result['reason'] = "Success"
    get_model_result['next_ts'] = 0
    return get_model_result


if __name__ == '__main__':
    while True:
        result, current_iteration, rsp_feature_maps, compress_type, sparse_rate = start_fl_job()
        sys.stdout.flush()
        if result['reason'] == "Restart iteration.":
            current_ts = datetime_to_timestamp(datetime.datetime.now())
            duration = result['next_ts'] - current_ts
            if duration >= 0:
                time.sleep(duration / 1000)
            continue

        result = update_model(current_iteration, rsp_feature_maps, compress_type, sparse_rate)
        sys.stdout.flush()
        if result['reason'] == "Restart iteration.":
            current_ts = datetime_to_timestamp(datetime.datetime.now())
            duration = result['next_ts'] - current_ts
            if duration >= 0:
                time.sleep(duration / 1000)
            continue

        result = get_result(current_iteration)
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
