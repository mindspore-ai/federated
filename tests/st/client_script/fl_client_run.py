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
"""start client"""

import os
import argparse
import shutil

parser = argparse.ArgumentParser(description="Run TestClient.java case")
parser.add_argument("--jarPath", type=str, default="mindspore-lite-java-flclient.jar")  # must be absolute path
parser.add_argument("--case_jarPath", type=str, default="case_jar/flclient_models.jar")  # must be absolute path
parser.add_argument("--train_dataset", type=str, default="client/train.txt")  # must be absolute path
parser.add_argument("--test_dataset", type=str, default="client/eval.txt")  # must be absolute path
parser.add_argument("--vocal_file", type=str, default="client/vocab.txt")  # must be absolute path
parser.add_argument("--ids_file", type=str, default="client/vocab_map_ids.txt")  # must be absolute path
parser.add_argument("--path_regex", type=str, default=",")

parser.add_argument("--flName", type=str, default="com.mindspore.flclient.demo.adbert.AdBertClient")

parser.add_argument("--train_model_path", type=str,
                    default="client/train/albert_ad_train.mindir.ms")  # must be absolute path of .ms files
parser.add_argument("--infer_model_path", type=str,
                    default="client/train/albert_ad_infer.mindir.ms")  # must be absolute path of .ms files

parser.add_argument("--ssl_protocol", type=str, default="TLSv1.2")
parser.add_argument("--deploy_env", type=str, default="x86")
parser.add_argument("--domain_name", type=str, default="https://10.113.216.106:6668")
parser.add_argument("--cert_path", type=str, default="certs/https_signature_certificate/client/CARoot.pem")
parser.add_argument("--use_elb", type=str, default="false")
parser.add_argument("--server_num", type=int, default=1)
parser.add_argument("--task", type=str, default="train")
parser.add_argument("--thread_num", type=int, default=1)
parser.add_argument("--cpu_bind_mode", type=str, default="NOT_BINDING_CORE")

parser.add_argument("--train_weight_name", type=str, default="null")
parser.add_argument("--infer_weight_name", type=str, default="null")
parser.add_argument("--name_regex", type=str, default=",")
parser.add_argument("--server_mode", type=str, default="FEDERATED_LEARNING")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--input_shape", type=str, default="null")

parser.add_argument("--client_num", type=int, default=0)

args, _ = parser.parse_known_args()

jar_path = args.jarPath
case_jar_path = args.case_jarPath

train_dataset = args.train_dataset
test_dataset = args.test_dataset
vocal_file = args.vocal_file
ids_file = args.ids_file
path_regex = args.path_regex

fl_name = args.flName

train_model_path = args.train_model_path
infer_model_path = args.infer_model_path

ssl_protocol = args.ssl_protocol
deploy_env = args.deploy_env
domain_name = args.domain_name
cert_path = args.cert_path
use_elb = args.use_elb
server_num = args.server_num
task = args.task
thread_num = args.thread_num
cpu_bind_mode = args.cpu_bind_mode

train_weight_name = args.train_weight_name
infer_weight_name = args.infer_weight_name
name_regex = args.name_regex
server_mode = args.server_mode
batch_size = args.batch_size
input_shape = args.input_shape

client_num = args.client_num


def get_client_data_path(user_path):
    """
    concat files of user_path
    Args:
        user_path: the data set file path
    """
    bin_file_paths = os.listdir(user_path)
    train_data_path = ""
    train_label_path = ""

    test_data_path = ""
    test_label_path = ""
    for file in bin_file_paths:
        info = file.split(".")[0].split("_")
        if info[4] == "train" and info[5] == "data":
            train_data_path = os.path.join(user_path, file)
        elif info[4] == "train" and info[5] == "label":
            train_label_path = os.path.join(user_path, file)
        elif info[4] == "test" and info[5] == "data":
            test_data_path = os.path.join(user_path, file)
        elif info[4] == "test" and info[5] == "label":
            test_label_path = os.path.join(user_path, file)
    train_data_label = train_data_path + "," + train_label_path
    test_path = test_data_path + "," + test_label_path

    return train_data_label, test_path, test_path


for i in range(client_num):
    fl_id = "f" + str(i)
    train_path, eval_path, infer_path = "", "", ""
    if "AlbertClient" in fl_name:
        print("AlBertClient")
        train_path = train_dataset + "," + vocal_file + "," + ids_file
        eval_path = test_dataset + "," + vocal_file + "," + ids_file
        infer_path = test_dataset + "," + vocal_file + "," + ids_file
    elif "LenetClient" in fl_name:
        print("LenetClient")
        train_path, eval_path, infer_path = get_client_data_path(train_dataset)
    elif "AdBertClient" in fl_name:
        print("AdBertClient")
        train_path = train_dataset + "," + vocal_file + "," + ids_file
        eval_path = test_dataset + "," + vocal_file + "," + ids_file
        infer_path = test_dataset + "," + vocal_file + "," + ids_file
    elif "VaeClient" in fl_name:
        print("VaeClient")
        train_path = train_dataset
        eval_path = train_dataset
        infer_path = train_dataset
    elif "TagClient" in fl_name:
        print("TagClient")
        train_path = train_dataset
        eval_path = test_dataset
        infer_path = test_dataset
    else:
        print("the flname is error")
    print("===========================")
    print("fl id: ", fl_id)
    print("train path: ", train_path)
    print("eval path: ", eval_path)
    print("infer path: ", infer_path)
    cmd_client = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && "
    cmd_client += "rm -rf ${execute_path}/client_" + task + str(i) + "/ &&"
    cmd_client += "mkdir ${execute_path}/client_" + task + str(i) + "/ &&"
    cmd_client += "cd ${execute_path}/client_" + task + str(i) + "/ || exit &&"

    MAIN_CLASS_NAME = "com.mindspore.flclient.SyncFLJob"
    java_class_path = "-cp " + case_jar_path + ":" + jar_path
    cmd_client += "java " + java_class_path + " "
    cmd_client += MAIN_CLASS_NAME + " "
    cmd_client += train_path + " "
    cmd_client += eval_path + " "
    cmd_client += infer_path + " "
    cmd_client += path_regex + " "
    cmd_client += fl_name + " "

    train_model_file = os.path.basename(train_model_path)
    train_model_dir = os.path.dirname(train_model_path)
    train_model_path_spec = os.path.join(train_model_dir, "test", str(i) + train_model_file)
    if os.path.exists(train_model_path_spec):
        os.remove(train_model_path_spec)
    shutil.copyfile(train_model_path, train_model_path_spec)
    cmd_client += train_model_path_spec + " "
    print("train model path: ", train_model_path_spec)

    infer_model_file = os.path.basename(infer_model_path)
    infer_model_dir = os.path.dirname(infer_model_path)
    infer_model_path_spec = os.path.join(infer_model_dir, "test", str(i) + infer_model_file)
    if os.path.exists(infer_model_path_spec):
        os.remove(infer_model_path_spec)
    shutil.copyfile(infer_model_path, infer_model_path_spec)
    cmd_client += infer_model_path_spec + " "
    print("infer model path: ", infer_model_path_spec)

    cmd_client += ssl_protocol + " "
    cmd_client += deploy_env + " "
    cmd_client += domain_name + " "
    cmd_client += cert_path + " "
    cmd_client += use_elb + " "
    cmd_client += str(server_num) + " "
    cmd_client += task + " "
    cmd_client += str(thread_num) + " "
    cmd_client += cpu_bind_mode + " "
    cmd_client += train_weight_name + " "
    cmd_client += infer_weight_name + " "
    cmd_client += name_regex + " "
    cmd_client += server_mode + " "
    cmd_client += str(batch_size) + " "
    cmd_client += input_shape + " "
    cmd_client += " > client-" + task + ".log 2>&1 &"
    print(cmd_client)
    os.system(cmd_client)
