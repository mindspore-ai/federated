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
"""Run client x86"""

import argparse
import subprocess
import os
import time


def copy_ms(origin_train_model_path, origin_infer_model_path, client_num, train_model_dir, infer_model_dir):
    """copy ms file"""
    if origin_train_model_path != "null":
        for i in range(client_num):
            if not os.path.exists(train_model_dir):
                os.makedirs(train_model_dir)
            os.system("cp -rf {} {}/train_{}.ms".format(origin_train_model_path, train_model_dir, i))
    if origin_infer_model_path != "null":
        if not os.path.exists(infer_model_dir):
            os.makedirs(infer_model_dir)
        for i in range(client_num):
            os.system("cp -rf {} {}/infer_{}.ms".format(origin_infer_model_path, infer_model_dir, i))
    time.sleep(1.0)


class DataPathConstructor:
    """get data path"""

    def __init__(self, fl_name, train_data_dir, eval_data_dir, infer_data_dir, vocab_path="null", ids_path="null"):
        self.train_path_list = list()
        self.eval_path_list = list()
        self.infer_path_list = list()
        if fl_name == "com.mindspore.flclient.demo.lenet.LenetClient":
            self.set_path_value(train_data_dir)
        elif fl_name == "com.mindspore.flclient.demo.albert.AlbertClient":
            file_name_list = ["{}.txt".format(_) for _ in range(20)]
            for file_name in file_name_list:
                self.train_path_list.append(
                    "{},{},{}".format(os.path.join(train_data_dir, file_name), vocab_path, ids_path))
            eval_data_path = os.path.join(eval_data_dir, "eval.txt")
            self.eval_path_list = ["{},{},{}".format(eval_data_path, vocab_path, ids_path) for _ in
                                   file_name_list] if eval_data_dir != "null" else ["null" for _ in file_name_list]
            infer_data_path = os.path.join(infer_data_dir, "eval.txt")
            self.infer_path_list = ["{},{},{}".format(infer_data_path, vocab_path, ids_path) for _ in
                                    file_name_list] if infer_data_dir != "null" else ["null" for _ in file_name_list]
        else:
            raise ValueError("you must check the fl_name")

    def set_path_value(self, train_data_dir):
        """set path value"""
        relative_path_list = os.listdir(train_data_dir)
        print(train_data_dir)
        for relative_path in relative_path_list:
            use_path = os.path.join(train_data_dir, relative_path)
            bin_file_paths = os.listdir(use_path)
            train_data_path, train_label_path, test_data_path, test_label_path = "", "", "", ""
            for file in bin_file_paths:
                info = file.split(".")[0].split("_")
                if info[4] == "train" and info[5] == "data":
                    train_data_path = os.path.join(use_path, file)
                elif info[4] == "train" and info[5] == "label":
                    train_label_path = os.path.join(use_path, file)
                elif info[4] == "test" and info[5] == "data":
                    test_data_path = os.path.join(use_path, file)
                elif info[4] == "test" and info[5] == "label":
                    test_label_path = os.path.join(use_path, file)
                else:
                    raise ValueError("you must check the data dir")
            self.train_path_list.append(train_data_path + "," + train_label_path)
            self.eval_path_list.append(test_data_path + "," + test_label_path)
            self.infer_path_list.append(test_data_path + "," + test_label_path)

    def get_train_path(self, index):
        return self.train_path_list[index]

    def get_eval_path(self, index):
        return self.eval_path_list[index]

    def get_infer_path(self, index):
        return self.infer_path_list[index]

def build_args_parser():
    """args parser"""
    parser = argparse.ArgumentParser(description="Run SyncFLJob.java case")
    parser.add_argument("--fl_jar_path", type=str, default="./fl_jar/minspore-lite-java-flclient.jar")
    parser.add_argument("--case_jar_path", type=str, default="./case_jar/quick-start-flclient.jar")
    parser.add_argument("--lite_jar_path", type=str, default="./case_jar/mindspore-lite-java.jar")  # must be absolute path
    parser.add_argument("--train_data_dir", type=str, default="../../datasets/3500_client_bin/")
    parser.add_argument("--eval_data_dir", type=str, default="null")
    parser.add_argument("--infer_data_dir", type=str, default="null")
    parser.add_argument("--vocab_path", type=str, default="")
    parser.add_argument("--ids_path", type=str, default="")
    parser.add_argument("--path_regex", type=str, default=",")
    parser.add_argument("--fl_name", type=str, default="com.mindspore.flclient.demo.lenet.LenetClient")
    parser.add_argument("--origin_train_model_path", type=str,
                        default="../../ms_files/lenet/lenet_train.ms")
    parser.add_argument("--origin_infer_model_path", type=str, default="")
    parser.add_argument("--train_model_dir", type=str, default="./ms/")
    parser.add_argument("--infer_model_dir", type=str, default="./ms/")
    parser.add_argument("--ssl_protocol", type=str, default="TLSv1.2")
    parser.add_argument("--deploy_env", type=str, default="x86")
    parser.add_argument("--domain_name", type=str, default="http://127.0.0.1:6666")
    parser.add_argument("--cert_path", type=str, default="./CARoot.pem")
    parser.add_argument("--use_elb", type=str, default="false")
    parser.add_argument("--server_num", type=int, default=1)
    parser.add_argument("--task", type=str, default="train")
    parser.add_argument("--thread_num", type=int, default=1)
    parser.add_argument("--cpu_bind_mode", type=str, default="NOT_BINDING_CORE")
    parser.add_argument("--train_weight_name", type=str, default="null")
    parser.add_argument("--infer_weight_name", type=str, default="null")
    parser.add_argument("--name_regex", type=str, default=",")
    parser.add_argument("--server_mode", type=str, default="FEDERATED_LEARNING")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_shape", type=str, default="null")
    parser.add_argument("--client_num", type=int, default=8)
    return parser

def main():
    parser = build_args_parser()
    args, _ = parser.parse_known_args()

    fl_jar_path = os.path.abspath(args.fl_jar_path)
    case_jar_path = os.path.abspath(args.case_jar_path)
    lite_jar_path = os.path.abspath(args.lite_jar_path)
    main_class_name = "com.mindspore.flclient.SyncFLJob"
    train_data_dir = args.train_data_dir if args.train_data_dir == "null" else os.path.abspath(args.train_data_dir)
    eval_data_dir = args.eval_data_dir if args.eval_data_dir == "null" else os.path.abspath(args.eval_data_dir)
    infer_data_dir = args.infer_data_dir if args.infer_data_dir == "null" else os.path.abspath(args.infer_data_dir)
    vocab_path = args.vocab_path if args.vocab_path == "null" else os.path.abspath(args.vocab_path)
    ids_path = args.ids_path if args.ids_path == "null" else os.path.abspath(args.ids_path)
    path_regex = args.path_regex
    fl_name = args.fl_name
    if args.origin_train_model_path == "null":
        origin_train_model_path = args.origin_train_model_path
    else:
        origin_train_model_path = os.path.abspath(args.origin_train_model_path)

    if args.origin_infer_model_path == "null":
        origin_infer_model_path = args.origin_infer_model_path
    else:
        origin_infer_model_path = os.path.abspath(args.origin_infer_model_path)
    train_model_dir = args.train_model_dir if args.train_model_dir == "null" else os.path.abspath(args.train_model_dir)
    infer_model_dir = args.infer_model_dir if args.infer_model_dir == "null" else os.path.abspath(args.infer_model_dir)
    ssl_protocol = args.ssl_protocol
    deploy_env = args.deploy_env
    domain_name = args.domain_name
    cert_path = args.cert_path if args.cert_path == "null" else os.path.abspath(args.cert_path)
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
    copy_ms(origin_train_model_path, origin_infer_model_path, client_num, train_model_dir, infer_model_dir)
    data_path_list = DataPathConstructor(fl_name, train_data_dir, eval_data_dir, infer_data_dir, vocab_path, ids_path)
    for i in range(client_num):
        cmd_client = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && "
        cmd_client += "rm -rf ${execute_path}/client_" + str(i) + "/ &&"
        cmd_client += "mkdir ${execute_path}/client_" + str(i) + "/ &&"
        cmd_client += "cd ${execute_path}/client_" + str(i) + "/ || exit &&"
        cmd_client += "java -Xms2048m -Xmx2048m -XX:+UseG1GC -cp {}:{}:{} {} ". \
            format(lite_jar_path, case_jar_path, fl_jar_path, main_class_name)
        train_path = data_path_list.get_train_path(i)
        eval_path = data_path_list.get_eval_path(i)
        infer_path = data_path_list.get_infer_path(i)
        cmd_client += train_path + " "
        cmd_client += eval_path + " "
        cmd_client += infer_path + " "
        cmd_client += path_regex + " "
        cmd_client += fl_name + " "
        cmd_client += os.path.join(train_model_dir, "train_{}.ms ".format(i))
        infer_model_path = os.path.join(train_model_dir, "train_{}.ms ".format(
            i)) if origin_infer_model_path == "null" else os.path.join(infer_model_dir, "infer_{}.ms ".format(i))
        cmd_client += infer_model_path
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
        cmd_client += " > client" + ".log 2>&1 &"
        print(cmd_client)
        subprocess.call(['bash', '-c', cmd_client])
if __name__ == "__main__":
    main()
