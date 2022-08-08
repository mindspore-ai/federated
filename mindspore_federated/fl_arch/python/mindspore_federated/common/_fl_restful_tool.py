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
"""
Function:
    Use to control the federated learning cluster
Usage:
    python _fl_restful_tool.py [http_type] [ip] [port] [request_name] [server_num] [instance_param] [metrics_file_path]
"""
import argparse
import json
import os
import warnings
from enum import Enum
import requests


class Status(Enum):
    """
    Response Status
    """
    SUCCESS = "0"
    FAILED = "1"


class Restful(Enum):
    """
    Define restful interface constant
    """
    GET_INSTANCE_DETAIL = "getInstanceDetail"
    NEW_INSTANCE = "newInstance"
    QUERY_INSTANCE = "queryInstance"
    ENABLE_FLS = "enableFLS"
    DISABLE_FLS = "disableFLS"
    STATE = "state"


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--http_type", type=str, default="http", help="http or https")
parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=6666)
parser.add_argument("--request_name", type=str, default="")

parser.add_argument("--server_num", type=int, default=0)
parser.add_argument("--instance_param", type=str, default="")
parser.add_argument("--metrics_file_path", type=str, default="/opt/huawei/mindspore/hybrid_albert/metrics.json")

args, _ = parser.parse_known_args()
http_type = args.http_type
ip = args.ip
port = args.port
request_name = args.request_name
server_num = args.server_num
instance_param = args.instance_param
metrics_file_path = args.metrics_file_path

headers = {'Content-Type': 'application/json'}
session = requests.Session()
base_url = http_type + "://" + ip + ":" + str(port) + "/"


def call_get_instance_detail():
    """
    get cluster instance detail
    """
    if not os.path.exists(metrics_file_path):
        return process_self_define_json(Status.FAILED.value, "error. metrics file is not existed.")

    ans_json_obj = {}

    with open(metrics_file_path, 'r') as f:
        metrics_list = f.readlines()

    if not metrics_list:
        return process_self_define_json(Status.FAILED.value, "error. metrics file has no content")

    last_metrics = metrics_list[len(metrics_list) - 1]
    last_metrics_obj = json.loads(last_metrics)

    ans_json_obj["code"] = Status.SUCCESS.value
    ans_json_obj["message"] = "get instance metrics detail successful."
    ans_json_obj["result"] = last_metrics_obj

    return json.dumps(ans_json_obj)


def call_new_instance():
    """
    call cluster new instance
    """
    if instance_param == "":
        return process_self_define_json(Status.FAILED.value, "error. instance_param is empty.")
    instance_param_list = instance_param.split(sep=",")
    instance_param_json_obj = {}

    url = base_url + Restful.NEW_INSTANCE.value
    for cur in instance_param_list:
        pair = cur.split(sep="=")
        if pair[0] == "update_model_ratio" or pair[0] == "client_learning_rate":
            instance_param_json_obj[pair[0]] = float(pair[1])
        else:
            instance_param_json_obj[pair[0]] = int(pair[1])

    data = json.dumps(instance_param_json_obj)
    res = session.post(url, verify=False, data=data)
    return json.loads(res.text)


def call_query_instance():
    """
    query cluster instance
    """
    url = base_url + Restful.QUERY_INSTANCE.value
    res = session.post(url, verify=False)
    return json.loads(res.text)


def call_enable_fls():
    """
    enable cluster fls
    """
    url = base_url + Restful.ENABLE_FLS.value
    res = session.post(url, verify=False)
    return json.loads(res.text)


def call_disable_fls():
    """
    disable cluster fls
    """
    url = base_url + Restful.DISABLE_FLS.value
    res = session.post(url, verify=False)
    return json.loads(res.text)


def call_state():
    """
    get cluster state
    """
    url = base_url + Restful.STATE.value
    res = session.get(url, verify=False)
    return json.loads(res.text)


def process_self_define_json(code, message):
    """
    process self define json
    """
    result_dict = {"code": code, "message": message}
    return json.dumps(result_dict)


if __name__ == '__main__':
    if request_name == Restful.GET_INSTANCE_DETAIL.value:
        print(call_get_instance_detail())

    elif request_name == Restful.NEW_INSTANCE.value:
        print(call_new_instance())

    elif request_name == Restful.QUERY_INSTANCE.value:
        print(call_query_instance())

    elif request_name == Restful.ENABLE_FLS.value:
        print(call_enable_fls())

    elif request_name == Restful.DISABLE_FLS.value:
        print(call_disable_fls())

    elif request_name == Restful.STATE.value:
        print(call_state())

    else:
        print("error. request_name is not found!")
