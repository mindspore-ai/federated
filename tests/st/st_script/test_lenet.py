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
"""test case for Lenet"""
import inspect
import os
import shutil
import pytest
from base_case import BaseCase

FLNAME = "com.mindspore.flclient.demo.lenet.LenetClient"


@pytest.mark.fl_cluster
class TestLenetTrain(BaseCase):
    """
    ST test class for Lenet train
    """
    train_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/lenet_train.ms")
    infer_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/lenet_train.ms")
    train_dataset = os.path.join(BaseCase.fl_resource_path, "client/data/lenet")
    test_dataset = os.path.join(BaseCase.fl_resource_path, "client/data/lenet")
    origin_ckpt_path = os.path.join(BaseCase.fl_resource_path, "server/models/lenet/fl_ckpt/")
    cur_ckpt_path = os.path.join(BaseCase.script_path, "lenet_cur_fl_ckpt/")

    def setup_method(self):
        """
        Run before every test case
        :return:
        """
        self.init_env("../cross_device_cloud", "quick_start_flclient.jar")
        self.stop_client()
        self.stop_cluster()
        self.start_redis()
        if os.path.exists(self.cur_ckpt_path):
            shutil.rmtree(self.cur_ckpt_path)
        shutil.copytree(self.origin_ckpt_path, self.cur_ckpt_path)

    def teardown_method(self):
        """
        Run after every test case
        :return:
        """
        self.stop_client()
        self.stop_cluster()
        self.clear_logs()

    def start_client(self):
        """
        start federated client
        """
        start_client_cmd = "cd {}/../client_script ;LD_LIBRARY_PATH={} python fl_client_run.py --jarPath={}  " \
                           "--case_jarPath={} --lite_jarPath={} --train_dataset={} --test_dataset={} --vocal_file={} " \
                           "--ids_file={} --flName={} --train_model_path={} --infer_model_path={} " \
                           "--ssl_protocol={}  --deploy_env={} --domain_name={} --cert_path={} " \
                           "--server_num={} --client_num={} --use_elb={} --thread_num={} --server_mode={} " \
                           "--batch_size={} --task={}" \
            .format(self.server_path, self.ld_library_path, self.frame_jar_path, self.case_jar_path, self.lite_jar_path,
                    self.train_dataset, "null", "null", "null", FLNAME, self.train_model_path,
                    self.infer_model_path, self.ssl_protocol, self.deploy_env, self.domain_name,
                    self.cert_path, self.server_num, self.client_num, self.use_elb, self.thread_num,
                    self.server_mode, self.client_batch_size, "train")
        print("exec:{}".format(start_client_cmd), flush=True)
        os.system(start_client_cmd)

    def check_client_log(self):
        # check client result
        query_success_cmd = "grep -E 'the total response of .: SUCCESS' {}/../client_script/client_train0/* |wc -l" \
            .format(self.server_path)
        print("query_success_cmd:" + query_success_cmd)
        result = os.popen(query_success_cmd)
        info = result.read()
        result.close()
        success_flg = int(info) >= 1
        if not success_flg:
            os.system("cat {}/../client_script/client_train0/*".format(self.server_path))
        assert success_flg is True

        # check acc not none
        query_acc_cmd = "grep 'evaluate acc' {}/../client_script/client_train0/* |wc -l".format(self.server_path)
        print("query_acc_cmd:" + query_acc_cmd)
        result = os.popen(query_acc_cmd)
        info = result.read()
        result.close()
        success_flg = info.find('none') == -1
        if not success_flg:
            os.system("cat {}/../client_script/client_train0/*".format(self.server_path))
        assert success_flg is True
        return True

    def test_train_lenet_nc_ne(self):
        """
        Feature: FL train process
        Description: test train lenet no compress, no encrypt
        Expectation: train success
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.client_num = 1
        self.start_scheduler("yamls/lenet/nc_ne_config.yaml")
        self.start_server("yamls/lenet/nc_ne_config.yaml", self.cur_ckpt_path)
        self.wait_cluster_ready(out_time=30)
        self.start_client()
        self.check_client_result(out_time=60)

    def test_train_lenet_compress_ne(self):
        """
        Feature: FL train process
        Description: test train lenet with compress, no encrypt
        Expectation: train success
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.client_num = 2
        self.start_scheduler("yamls/lenet/compress_ne_config.yaml")
        self.start_server("yamls/lenet/compress_ne_config.yaml", self.cur_ckpt_path)
        self.wait_cluster_ready(out_time=30)
        self.start_client()
        self.check_client_result(out_time=60)

    def test_train_lenet_nc_dp(self):
        """
        Feature: FL train process
        Description: test train lenet with no compress, dp encrypt
        Expectation: train success
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.client_num = 3
        self.start_scheduler("yamls/lenet/nc_dp_config.yaml")
        self.start_server("yamls/lenet/nc_dp_config.yaml", self.cur_ckpt_path)
        self.wait_cluster_ready(out_time=30)
        self.start_client()
        self.check_client_result(out_time=300)

    def test_train_lenet_nc_pw(self):
        """
        Feature: FL train process
        Description: test train lenet with no compress, pw encrypt
        Expectation: train success
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.client_num = 4
        self.start_scheduler("yamls/lenet/nc_pw_config.yaml")
        self.start_server("yamls/lenet/nc_pw_config.yaml", self.cur_ckpt_path)
        self.wait_cluster_ready(out_time=30)
        self.start_client()
        self.check_client_result(out_time=300)

    def test_train_lenet_nc_signds(self):
        """
        Feature: FL train process
        Description: test train lenet with no compress, signds encrypt
        Expectation: train success
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.client_num = 4
        self.start_scheduler("yamls/lenet/nc_signds_config.yaml")
        self.start_server("yamls/lenet/nc_signds_config.yaml", self.cur_ckpt_path)
        self.wait_cluster_ready(out_time=30)
        self.start_client()
        self.check_client_result(out_time=300)


@pytest.mark.fl_cluster
class TestLenetInference(BaseCase):
    """
    ST test class for Lenet infer
    """
    train_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/lenet_train.ms")
    infer_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/lenet_train.ms")
    train_dataset = os.path.join(BaseCase.fl_resource_path, "client/data/lenet")
    test_dataset = os.path.join(BaseCase.fl_resource_path, "client/data/lenet")

    def setup_method(self):
        """
        Run before every test case
        :return:
        """
        self.init_env("../cross_device_cloud", "quick_start_flclient.jar")
        finish_client_cmd = "cd {}/../client_script; " \
                            "python fl_client_finish.py --kill_tag=mindspore-lite-java-flclient" \
            .format(self.server_path)
        os.system(finish_client_cmd)

    def teardown_method(self):
        """
        Run after every test case
        :return:
        """
        finish_client_cmd = "cd {}/../client_script; " \
                            "python fl_client_finish.py --kill_tag=mindspore-lite-java-flclient" \
            .format(self.server_path)
        os.system(finish_client_cmd)
        self.clear_logs()

    def start_client(self):
        """
        start federated client
        """
        start_client_cmd = "cd {}/../client_script ;LD_LIBRARY_PATH={} python fl_client_run.py --jarPath={}  " \
                           "--case_jarPath={} --lite_jarPath={} --train_dataset={} --test_dataset={} --vocal_file={} " \
                           "--ids_file={} --flName={} --train_model_path={} --infer_model_path={} " \
                           "--ssl_protocol={}  --deploy_env={} --domain_name={} --cert_path={} " \
                           "--server_num={} --client_num={} --use_elb={} --thread_num={} --server_mode={} " \
                           "--batch_size={} --task={}" \
            .format(self.server_path, self.ld_library_path, self.frame_jar_path, self.case_jar_path, self.lite_jar_path,
                    self.train_dataset, self.test_dataset, "null", "null",
                    FLNAME, self.train_model_path, self.infer_model_path, self.ssl_protocol, self.deploy_env,
                    self.domain_name, self.cert_path, self.server_num, self.client_num, self.use_elb, self.thread_num,
                    self.server_mode, self.client_batch_size, "inference")
        print("exec:{}".format(start_client_cmd), flush=True)
        os.system(start_client_cmd)

    def check_client_log(self):
        """
        check client log to get test result
        :return:
        """
        # check client result
        query_success_cmd = "grep 'inference finish' {}/../client_script/client_inference0/* |wc -l".format(
            self.server_path)
        print("query_success_cmd:" + query_success_cmd)
        result = os.popen(query_success_cmd)
        info = result.read()
        result.close()
        assert int(info) == 1

    def test_infer_lenet(self):
        """
        infer case
        :return:
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.start_client()
        self.check_client_result(out_time=30)
