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
"""test case for vae"""
import inspect
import os
import shutil
import pytest
from base_case import BaseCase

FLNAME = "com.mindspore.flclient.demo.vae.VaeClient"


@pytest.mark.fl_cluster
class TestVaeTrain(BaseCase):
    """
    ST Test class for vae train
    """
    train_dataset = os.path.join(BaseCase.fl_resource_path,
                                 "client/data/vae/flatten_ca801543-a7e8-4090-9210-9b5af63be892_3.csv")
    test_dataset = os.path.join(BaseCase.fl_resource_path,
                                "client/data/vae/flatten_ca801543-a7e8-4090-9210-9b5af63be892_3.csv")
    train_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/vae_train_2022.0411.ms")
    infer_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/vae_train_2022.0411.ms")
    origin_ckpt_path = os.path.join(BaseCase.fl_resource_path, "server/models/vae/fl_ckpt/")
    cur_ckpt_path = os.path.join(BaseCase.fl_resource_path, "server/models/vae/cur_fl_ckpt/")

    def setup_method(self):
        """
        Run before every test case
        :return:
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.init_env("../cross_device_cloud", "flclient_models.jar")
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
                           "--case_jarPath={} --train_dataset={} --test_dataset={} --vocal_file={} " \
                           "--ids_file={} --flName={} --train_model_path={} --infer_model_path={} " \
                           "--ssl_protocol={}  --deploy_env={} --domain_name={} --cert_path={} " \
                           "--server_num={} --client_num={} --use_elb={} --thread_num={} --server_mode={} " \
                           "--batch_size={} --task={}" \
            .format(self.server_path, self.ld_library_path, self.frame_jar_path, self.case_jar_path,
                    self.train_dataset, "null", "null", "null", FLNAME, self.train_model_path, self.infer_model_path,
                    self.ssl_protocol, self.deploy_env, self.domain_name, self.cert_path,
                    self.server_num, self.client_num, self.use_elb, self.thread_num, self.server_mode,
                    self.client_batch_size, "train")
        print("exec:{}".format(start_client_cmd), flush=True)
        os.system(start_client_cmd)

    def check_client_log(self):
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        # check client result
        query_success_cmd = "grep 'the total response of 1: SUCCESS' {}/../client_script/client_train0/* |wc -l".format(
            self.server_path)
        print("query_success_cmd:" + query_success_cmd)
        result = os.popen(query_success_cmd)
        info = result.read()
        result.close()
        success_flg = int(info) >= 1
        if not success_flg:
            os.system("cat {}/../client_script/client_train0/*".format(self.server_path))
        assert success_flg is True

        # check if nan exist
        query_nan_cmd = "grep 'is nan' {}/../client_script/client_train0/* |wc -l".format(self.server_path)
        result = os.popen(query_nan_cmd)
        info = result.read()
        result.close()
        # after refresh the resource change to int(info) == 0
        success_flg = int(info) == 0
        if not success_flg:
            os.system("cat {}/../client_script/client_train0/*".format(self.server_path))
        assert success_flg is True

    def test_train_vae_nc_ne(self):
        """
        Feature: FL train process
        Description: test train vae no compress, no encrypt
        Expectation: train success
        """
        self.client_num = 1
        self.start_scheduler("yamls/vae/nc_ne_config.yaml")
        self.start_server("yamls/vae/nc_ne_config.yaml", self.cur_ckpt_path)
        self.wait_cluster_ready(out_time=30)
        self.start_client()
        self.check_client_result(out_time=60)

    def test_train_vae_compress_ne(self):
        """
        Feature: FL train process
        Description: test train vae with compress, no encrypt
        Expectation: train success
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.client_num = 2
        self.start_scheduler("yamls/vae/compress_ne_config.yaml")
        self.start_server("yamls/vae/compress_ne_config.yaml", self.cur_ckpt_path)
        self.wait_cluster_ready(out_time=30)
        self.start_client()
        self.check_client_result(out_time=60)

    def test_train_vae_nc_dp(self):
        """
        Feature: FL train process
        Description: test train vae with no compress, dp encrypt
        Expectation: train success
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.client_num = 3
        self.start_scheduler("yamls/vae/nc_dp_config.yaml")
        self.start_server("yamls/vae/nc_dp_config.yaml", self.cur_ckpt_path)
        self.wait_cluster_ready(out_time=30)
        self.start_client()
        self.check_client_result(out_time=300)

    def test_train_vae_nc_pw(self):
        """
        Feature: FL train process
        Description: test train vae with no compress, pw encrypt
        Expectation: train success
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.client_num = 4
        self.start_scheduler("yamls/vae/nc_pw_config.yaml")
        self.start_server("yamls/vae/nc_pw_config.yaml", self.cur_ckpt_path)
        self.wait_cluster_ready(out_time=30)
        self.start_client()
        self.check_client_result(out_time=300)

    def test_train_vae_nc_signds(self):
        """
        Feature: FL train process
        Description: test train vae with no compress, signds encrypt
        Expectation: train success
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.client_num = 4
        self.start_scheduler("yamls/vae/nc_signds_config.yaml")
        self.start_server("yamls/vae/nc_signds_config.yaml", self.cur_ckpt_path)
        self.wait_cluster_ready(out_time=30)
        self.start_client()
        self.check_client_result(out_time=300)


@pytest.mark.fl_cluster
class TestVaeInference(BaseCase):
    """
    ST test class for Vae inference
    """
    train_dataset = os.path.join(BaseCase.fl_resource_path,
                                 "client/data/vae/flatten_ca801543-a7e8-4090-9210-9b5af63be892_3.csv")
    test_dataset = os.path.join(BaseCase.fl_resource_path,
                                "client/data/vae/flatten_ca801543-a7e8-4090-9210-9b5af63be892_3.csv")
    train_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/vae_train_2022.0411.ms")
    infer_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/vae_train_2022.0411.ms")

    def setup_method(self):
        """
        Run before every test case
        :return:
        """
        self.init_env("../cross_device_cloud", "flclient_models.jar")
        self.stop_client()

    def teardown_method(self):
        """
        Run after every test case
        :return:
        """
        self.stop_client()
        self.clear_logs()

    def start_client(self):
        """
        start federated client
        """
        start_client_cmd = "cd {}/../client_script ;LD_LIBRARY_PATH={} python fl_client_run.py --jarPath={}  " \
                           "--case_jarPath={} --train_dataset={} --test_dataset={} --vocal_file={} " \
                           "--ids_file={} --flName={} --train_model_path={} --infer_model_path={} " \
                           "--ssl_protocol={}  --deploy_env={} --domain_name={} --cert_path={} " \
                           "--server_num={} --client_num={} --use_elb={} --thread_num={} --server_mode={} " \
                           "--batch_size={} --task={}" \
            .format(self.server_path, self.ld_library_path, self.frame_jar_path, self.case_jar_path,
                    self.train_dataset, self.test_dataset, "null", "null", FLNAME, self.train_model_path,
                    self.infer_model_path, self.ssl_protocol, self.deploy_env, self.domain_name,
                    self.cert_path, self.server_num, self.client_num, self.use_elb, self.thread_num,
                    self.server_mode, self.client_batch_size, "inference")
        print("exec:{}".format(start_client_cmd), flush=True)
        os.system(start_client_cmd)

    def check_client_log(self):
        # check client result
        query_success_cmd = "grep 'inference finish' {}/../client_script/client_inference0/* |wc -l".format(
            self.server_path)
        print("query_success_cmd:" + query_success_cmd)
        result = os.popen(query_success_cmd)
        info = result.read()
        result.close()
        assert int(info) == 1

    def test_infer_vae(self):
        """
        fist case
        :return:
        """
        self.start_client()
        self.check_client_result(out_time=30)
