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
"""ST test Base class."""

import os
import socket
import inspect
import time
from abc import abstractmethod

FRAME_PKG_INIT_FLG = False


class BaseCase:
    """
    ST test Base class.
    """
    # Common ENV
    fl_resource_path = os.getenv("FL_RESOURCE_PATH")
    ms_lite_pkg = os.getenv("MS_LITE_PKG")
    fl_jdk_path = os.getenv("FL_JDK_PATH")
    ms_install_pkg = os.getenv("MS_INSTAll_PKG")
    fl_install_pkg = os.getenv("FL_INSTALL_PKG")
    fl_cli_frame_jar = os.getenv("FL_ClI_FRAME_JAR")

    # ENV for Server
    script_path, _ = os.path.split(os.path.realpath(__file__))
    server_path = ""
    config_file_path = ""
    temp_path = os.path.join(script_path, "temp")
    scheduler_mgr_port = 6000
    scheduler_port = 60001
    fl_server_port = 60002
    redis_server_port = 8113
    scheduler_ip = socket.gethostbyname(socket.gethostname())
    fl_server_ip = socket.gethostbyname(socket.gethostname())
    server_num = 1
    worker_num = 0

    # ENV for Client
    lite_lib_path = os.path.join(script_path, "libs")
    ld_library_path = ""
    frame_jar_path = os.path.join(script_path, "frame_jar", os.path.basename(fl_cli_frame_jar))
    case_jar_path = ""
    ssl_protocol = "TLSv1.2"
    deploy_env = "x86"
    domain_name = "http://{}:{}".format(scheduler_ip, fl_server_port)
    cert_path = os.path.join(fl_resource_path, "client/cert/CARoot.pem")
    client_batch_size = 32
    client_num = 1
    use_elb = "false"
    thread_num = 1
    server_mode = "FEDERATED_LEARNING"

    def init_frame_pkg(self, case_jar):
        """
        unpack mindspore-lite-version-linux-x64.tar.gz get libs for fl client
        :return:
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        global FRAME_PKG_INIT_FLG
        if FRAME_PKG_INIT_FLG is True:
            os.system('rm -rf {}/case_jar; mkdir -p {}/case_jar'.format(self.script_path, self.script_path))
            cp_case_jar = "cp {}/ci_jar/{} {}/case_jar".format(self.fl_resource_path, case_jar, self.script_path)
            os.system(cp_case_jar)
            return
        FRAME_PKG_INIT_FLG = True
        os.system('rm -rf {}'.format(self.case_jar_path))
        os.system('rm -rf {}'.format(self.frame_jar_path))

        if os.path.exists(self.temp_path):
            os.system('rm -rf {}/mindspore-lite*'.format(self.temp_path))
        os.system('mkdir -p {}'.format(self.temp_path))
        if os.path.exists(self.lite_lib_path):
            os.system('rm -rf {}'.format(self.lite_lib_path))
        os.system('mkdir -p {}'.format(self.lite_lib_path))
        unpack_cmd = "tar -zxf {} -C {}".format(self.ms_lite_pkg, self.temp_path)
        os.system(unpack_cmd)
        lib_cp_cmd = "cp {}/mindspore-lite*/runtime/lib/* {}".format(self.temp_path, self.lite_lib_path)
        os.system(lib_cp_cmd)
        lib_cp_cmd = "cp {}/mindspore-lite*/runtime/third_party/libjpeg-turbo/lib/* {}" \
            .format(self.temp_path, self.lite_lib_path)
        os.system(lib_cp_cmd)

        os.system('rm -rf {}/frame_jar; mkdir -p {}/frame_jar'.format(self.script_path, self.script_path))
        cp_frame_jar = "cp {} {}/frame_jar" \
            .format(self.fl_cli_frame_jar, self.script_path)
        os.system(cp_frame_jar)
        os.system('rm -rf {}/case_jar; mkdir -p {}/case_jar'.format(self.script_path, self.script_path))
        cp_case_jar = "cp {}/ci_jar/{} {}/case_jar".format(self.fl_resource_path, case_jar, self.script_path)
        os.system(cp_case_jar)

        install_ms_cmd = "pip3 install {}".format(self.ms_install_pkg)
        os.system(install_ms_cmd)
        install_fl_cmd = "pip3 install {}".format(self.fl_install_pkg)
        os.system(install_fl_cmd)

    def init_env(self, relative_model_path, case_jar):
        """
        :param relative_model_path: relative directory of model path
        :param case_jar: jar of case
        :return:
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)

        # ENV for Server
        self.server_path = os.path.realpath(os.path.join(self.script_path, relative_model_path))
        self.config_file_path = os.path.join(self.server_path, "config.json")

        # ENV for Client
        env_library_path = os.getenv("LD_LIBRARY_PATH", default="")
        self.ld_library_path = env_library_path + ':' + self.lite_lib_path
        self.case_jar_path = os.path.join(self.script_path, "case_jar", case_jar)

        # prepare pkg
        self.init_frame_pkg(case_jar)

    def stop_cluster(self):
        finish_cluster_cmd = "cd {}; python finish_cloud.py --redis_port={}" \
            .format(self.server_path, self.redis_server_port)
        os.system(finish_cluster_cmd)

    def stop_client(self):
        finish_client_cmd = "cd {}/../client_script; " \
                            "python fl_client_finish.py --kill_tag=mindspore-lite-java-flclient" \
            .format(self.server_path)
        os.system(finish_client_cmd)

    def start_redis(self):
        start_redis_cmd = 'redis-server --port {} --save "" &'.format(self.redis_server_port)
        os.system(start_redis_cmd)

    def after_finish(self):
        finish_client_cmd = "cd {}/../client_script; " \
                            "python fl_client_finish.py --kill_tag=mindspore-lite-java-flclient" \
            .format(self.server_path)
        finish_cluster_cmd = "cd {}; python finish_cloud.py --redis_port={}" \
            .format(self.server_path, self.redis_server_port)
        os.system(finish_cluster_cmd)
        os.system(finish_client_cmd)
        self.clear_logs()

    def start_scheduler(self, cfg_file="default_yaml_config.yaml"):
        """
        Start scheduler
        :return:
        """
        start_scheduler_cmd = "cd {}; python run_sched.py --yaml_config={} --scheduler_manage_address=127.0.0.1:{}" \
            .format(self.server_path, cfg_file, self.scheduler_mgr_port)
        print("exec:{}".format(start_scheduler_cmd), flush=True)
        os.system(start_scheduler_cmd)

    def start_server(self, cfg_file="default_yaml_config.yaml", ckpt_path="./fl_ckpt/"):
        start_server_cmd = "cd {}; python run_server.py --yaml_config={} " \
                           "--http_server_address={}:{} --checkpoint_dir={} " \
                           "--local_server_num={} --tcp_server_ip=127.0.0.1" \
            .format(self.server_path, cfg_file, self.fl_server_ip, self.fl_server_port,
                    ckpt_path, self.server_num)

        print("exec:{}".format(start_server_cmd), flush=True)
        os.system(start_server_cmd)

    def wait_client_exit(self, out_time):
        """
        Wait until client finish work or time out
        """
        # wait client exit
        query_state_cmd = "ps -ef|grep mindspore-lite-java-flclient |grep -v grep | wc -l"
        finish_flg = False
        loop_times = 0
        while loop_times < out_time:
            result = os.popen(query_state_cmd)
            info = result.read()
            result.close()
            if int(info) == 0:
                finish_flg = True
                break
            time.sleep(1)
            loop_times = loop_times + 1
        # print logs while exception
        if not finish_flg:
            os.system("cat {}/../client_script/client_train0/*".format(self.server_path))
            assert finish_flg is True

    def check_client_result(self, out_time):
        """
        get test result
        :return:
        """
        self.wait_client_exit(out_time)
        self.check_client_log()

    @abstractmethod
    def check_client_log(self):
        """
        check client log to get test result
        :return:
        """
        print("The subclass must impl check_client_log")
        assert False

    def wait_cluster_ready(self, out_time=30):
        """
        Wait until server cluster ready or time out
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        # wait server status to ready
        query_state_cmd = "curl -k http://127.0.0.1:{}/state".format(self.scheduler_mgr_port)

        ready_flg = False
        loop_times = 0
        while loop_times < out_time:
            result = os.popen(query_state_cmd)
            info = result.read()
            print(info)
            result.close()
            if info.find('CLUSTER_READY') != -1:
                ready_flg = True
                break
            time.sleep(1)
            loop_times = loop_times + 1
        # print logs while exception
        if not ready_flg:
            os.system("cat {}/logs/scheduler/scheduler.log".format(self.server_path))
            os.system("cat {}/logs/server_{}/server.log".format(self.server_path, self.fl_server_port))
            assert ready_flg

    def clear_logs(self):
        print("not delete for debug")
        clear_server_log = "rm -rf {}/logs/server_{}".format(self.server_path, self.fl_server_port)
        clear_scheduler_log = "rm -rf {}/logs/scheduler".format(self.server_path)
        clear_train_log = "rm -rf {}/../client_script/client_train*".format(self.server_path)
        clear_infer_log = "rm -rf {}/../client_script/client_inference*".format(self.server_path)
        os.system(clear_server_log)
        os.system(clear_scheduler_log)
        os.system(clear_train_log)
        os.system(clear_infer_log)
