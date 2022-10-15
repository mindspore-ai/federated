# pylint: disable=broad-except

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
"""common python file"""
import json
import logging
import os
import signal
import socket
import subprocess
import time
import traceback
from functools import wraps
from multiprocessing import Process, Pipe

import requests
import yaml
import numpy as np
import psutil

from mindspore_federated import FLServerJob, FlSchedulerJob, Callback
from mindspore_federated import log as logger

from common_client import ResponseCode, ResponseFLJob, ResponseGetModel
from common_client import post_start_fl_job, post_get_model, post_update_model
from common_client import server_safemode_rsp
from common_data import generate_random_data


def fl_test(func):
    """fl test"""
    def clean_temp_files():
        cwd_dir = os.getcwd()
        temp_dir = os.path.join(cwd_dir, "temp")
        os.system(f"rm -rf {temp_dir}")
        fl_ckpt_dir = os.path.join(cwd_dir, "fl_ckpt")
        os.system(f"rm -rf {fl_ckpt_dir}")
        os.system(f"rm -f metrics.json")
        os.system(f"rm -f event.txt")
        os.system(f"rm -f alice_check")
        os.system(f"rm -f alice_pba_bf")
        os.system(f"rm -f bob_align_result")
        os.system(f"rm -f bob_p_b")
        os.system(f"rm -f client_psi_init")
        os.system(f"rm -f server_psi_init")

    @wraps(func)
    def wrap_test(*args, **kwargs):
        try:
            recover_redis_server()
            clean_temp_files()
            try:
                os.mkdir("temp")
            except FileExistsError:
                pass
            try:
                os.mkdir("fl_ckpt")
            except FileExistsError:
                pass
            func(*args, **kwargs)
        except Exception:
            logger.error("FL test catch exception")
            temp_dir = os.path.join(os.getcwd(), "temp")
            os.system(f"ls -l {temp_dir}/*.yaml && cat {temp_dir}/*.yaml")
            raise
        finally:
            logger.info("Fl test begin to clear")
            # pylint: disable=used-before-assignment
            global g_server_processes
            stop_processes(g_server_processes)
            g_server_processes = []
            clean_temp_files()
            logger.info("Fl test end clear")

    return wrap_test


def vfl_data_test(func):
    """vfl data test"""

    def clean_temp_files():
        cwd_dir = os.getcwd()
        temp_dir = os.path.join(cwd_dir, "temp")
        os.system(f"rm -rf {temp_dir}")

    def mkdir(directory):
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

    @wraps(func)
    def wrap_test(*args, **kwargs):
        try:
            clean_temp_files()
            mkdir("temp")
            mkdir("temp/leader")
            mkdir("temp/follower")
            generate_random_data()
            func(*args, **kwargs)
        except Exception:
            logger.error("VFL data test catch exception")
            raise
        finally:
            logger.info("VFl data test begin to clear")
            clean_temp_files()
            logger.info("VFl data test end clear")

    return wrap_test


g_server_processes = []
g_redis_server_port = int(os.environ["REDIS_SERVER_PORT"])
g_redis_server_address = f"127.0.0.1:{g_redis_server_port}"

g_fl_name_idx = 1
fl_training_mode = "FEDERATED_LEARNING"
fl_hybrid_mode = "HYBRID_TRAINING"

# pylint: disable=unused-argument
def fl_name_with_idx(fl_name):
    global g_fl_name_idx
    new_fl_name = f"fl_name_{g_fl_name_idx}"
    g_fl_name_idx += 1
    return new_fl_name


def get_default_ssl_config():
    cert_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../fl_ssl_cert/"))
    print(f"cert_dir: {cert_dir}")
    server_cert_path = os.path.join(cert_dir, "server.p12")
    client_cert_path = os.path.join(cert_dir, "client.p12")
    ca_cert_path = os.path.join(cert_dir, "ca.crt")
    server_password = "server_password_12345"
    client_password = "client_password_12345"
    return server_cert_path, client_cert_path, ca_cert_path, server_password, client_password


def get_default_redis_ssl_config():
    cert_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../fl_redis_ssl_cert/"))
    print(f"fl_redis_ssl_cert: {cert_dir}")
    server_cert_path = os.path.join(cert_dir, "server.crt")
    server_key_path = os.path.join(cert_dir, "serverkey.pem")
    client_cert_path = os.path.join(cert_dir, "client.crt")
    client_key_path = os.path.join(cert_dir, "clientkey.pem")
    ca_cert_path = os.path.join(cert_dir, "ca.crt")
    return server_cert_path, server_key_path, client_cert_path, client_key_path, ca_cert_path


def make_yaml_config(fl_name, update_configs, output_yaml_file, enable_ssl=False,
                     server_mode=None, distributed_cache_address=None, fl_iteration_num=None,
                     start_fl_job_threshold=None, update_model_ratio=None,
                     start_fl_job_time_window=None, update_model_time_window=None, global_iteration_time_window=None,
                     pki_verify=None, rmv_configs=None):
    """make yaml config"""
    if rmv_configs is None:
        rmv_configs = []
    cur_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(cur_dir, "default_yaml_config.yaml")) as fp:
        yaml_file_content = fp.read()
    yaml_configs = yaml.load(yaml_file_content, yaml.Loader)

    # convert multi layer dict to one layer
    def multi_layer_dict_as_one_layer(dst_dict, prefix, input_dict):
        for key, val in input_dict.items():
            if isinstance(val, dict):
                multi_layer_dict_as_one_layer(dst_dict, prefix + key + ".", val)
            else:
                dst_dict[prefix + key] = val
    yaml_configs_new = {}
    multi_layer_dict_as_one_layer(yaml_configs_new, "", yaml_configs)
    yaml_configs = yaml_configs_new

    update_configs["fl_name"] = fl_name

    def set_when_not_none(dst, key, val):
        if val is not None:
            dst[key] = val

    set_when_not_none(update_configs, "server_mode", server_mode)
    set_when_not_none(update_configs, "fl_iteration_num", fl_iteration_num)
    set_when_not_none(update_configs, "enable_ssl", enable_ssl)
    set_when_not_none(update_configs, "round.start_fl_job_threshold", start_fl_job_threshold)
    set_when_not_none(update_configs, "round.update_model_ratio", update_model_ratio)
    set_when_not_none(update_configs, "round.start_fl_job_time_window", start_fl_job_time_window)
    set_when_not_none(update_configs, "round.update_model_time_window", update_model_time_window)
    set_when_not_none(update_configs, "round.global_iteration_time_window", global_iteration_time_window)

    if pki_verify is not None:
        update_configs["client_verify.pki_verify"] = pki_verify
        update_configs["client_verify.root_first_ca_path"] = "xxxx"
        update_configs["client_verify.root_second_ca_path"] = "xxxx"
        update_configs["client_verify.equip_crl_path"] = "xxxx"

    server_cert_path, client_cert_path, ca_cert_path, _, _ = get_default_ssl_config()
    if "ssl.server_cert_path" not in update_configs:
        update_configs["ssl.server_cert_path"] = server_cert_path
    if "ssl.client_cert_path" not in update_configs:
        update_configs["ssl.client_cert_path"] = client_cert_path
    if "ssl.ca_cert_path" not in update_configs:
        update_configs["ssl.ca_cert_path"] = ca_cert_path

    set_when_not_none(update_configs, "distributed_cache.address", distributed_cache_address)

    for key, val in update_configs.items():
        yaml_configs[key] = val

    rmv_list_real = []
    for key, val in yaml_configs.items():
        if any(key == item or key[:len(item)+1] == item + "." for item in rmv_configs):
            rmv_list_real.append(key)
    for item in rmv_list_real:
        yaml_configs.pop(item)

    # convert one layer dict to multi layer
    yaml_configs_new = {}
    for key, val in yaml_configs.items():
        keys = key.split(".")
        cur = yaml_configs_new
        for i in range(len(keys)):
            item = keys[i]
            if i == len(keys) - 1:
                cur[item] = val
            else:
                if item not in cur:
                    cur[item] = {}
                cur = cur[item]
    yaml_configs = yaml_configs_new
    new_yaml_content = yaml.dump(yaml_configs, Dumper=yaml.Dumper)
    with open(output_yaml_file, "w") as fp:
        fp.write(new_yaml_content)


def start_fl_server(feature_map, yaml_config, http_server_address, tcp_server_ip="127.0.0.1",
                    checkpoint_dir="./fl_ckpt/", ssl_config=None, max_time_sec_wait=10):
    """start fl server"""
    print("new server process", flush=True)
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    send_pipe, recv_pipe = Pipe()

    class FlCallback(Callback):
        def after_started(self):
            send_pipe.send("Success")

    callback = FlCallback()

    def server_process_fun():
        try:
            server_job = FLServerJob(yaml_config, http_server_address, tcp_server_ip, checkpoint_dir,
                                     ssl_config=ssl_config)
            server_job.run(feature_map, callback=callback)
        except Exception as e:
            traceback.print_exc()
            logging.exception(e)
            send_pipe.send(e)

    server_process = Process(target=server_process_fun, args=tuple())
    server_process.start()
    global g_server_processes
    g_server_processes.append(server_process)
    index = 0
    while index < max_time_sec_wait * 10:  # wait max max_time_sec_wait s
        index += 1
        if recv_pipe.poll(0.1):
            msg = recv_pipe.recv()
            print(f"Receive server process msg: {msg} {server_process.is_alive()}")
            if isinstance(msg, Exception):
                raise msg
            break
        try:
            process = psutil.Process(server_process.pid)
            assert process.is_running()
        except psutil.NoSuchProcess:
            pass
    assert index < max_time_sec_wait * 10
    return server_process


def start_fl_scheduler(yaml_config, scheduler_http_address):
    """start fl scheduler"""
    print("new scheduler process", flush=True)
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    send_pipe, recv_pipe = Pipe()

    def server_process_fun():
        try:
            server_job = FlSchedulerJob(yaml_config, manage_address=scheduler_http_address)
            server_job.run()
            send_pipe.send("Success")
        except Exception as e:
            traceback.print_exc()
            logging.exception(e)
            send_pipe.send(e)

    scheduler_process = Process(target=server_process_fun, args=tuple())
    scheduler_process.start()
    global g_server_processes
    g_server_processes.append(scheduler_process)
    index = 0
    conn_server = socket.socket()

    http_ip = None
    scheduler_http_port = None
    try:
        address_split = scheduler_http_address.split(":")
        if len(address_split) == 2:
            http_ip = address_split[0]
            scheduler_http_port = int(address_split[1])
    except Exception:
        http_ip = None
        scheduler_http_port = None

    while index < 100:  # wait max 10 s
        index += 1
        if recv_pipe.poll(0.1):
            msg = recv_pipe.recv()
            print(f"Receive server process msg: {msg} {scheduler_process.is_alive()}")
            if isinstance(msg, Exception):
                raise msg
            break
        try:
            process = psutil.Process(scheduler_process.pid)
            assert process.is_running()
        except psutil.NoSuchProcess:
            pass
        if http_ip and scheduler_http_port:
            try:
                conn_server.connect((http_ip, scheduler_http_port))
                break
            except socket.error:
                pass
    assert index < 100
    return scheduler_process


def run_worker_client_task(task):
    """run worker client task"""
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    send_pipe, recv_pipe = Pipe()

    def worker_task_fun():
        try:
            task()
            send_pipe.send("Success")
        except Exception as e:
            traceback.print_exc()
            logging.exception(e)
            send_pipe.send(e)

    worker_task_process = Process(target=worker_task_fun, args=tuple())
    worker_task_process.start()
    global g_server_processes
    g_server_processes.append(worker_task_process)
    return worker_task_process, recv_pipe


def wait_worker_client_task_result(worker_task_process, recv_pipe, max_run_secs=3):
    """wait worker client task result"""
    index = 0
    while index < max_run_secs * 2:
        index += 1
        if recv_pipe.poll(0.1):
            msg = recv_pipe.recv()
            print(f"Receive server process msg: {msg} {worker_task_process.is_alive()}")
            if isinstance(msg, Exception):
                raise msg
            break
        time.sleep(0.5)

    assert index < max_run_secs * 2


def stop_processes(server_processes):
    """Stop processes. return True if stopped by terminate signal, return False if stopped by kill signal"""
    if not isinstance(server_processes, (list, tuple)):
        server_processes = [server_processes]
    server_processes2 = []
    for process in server_processes:
        try:
            process = psutil.Process(process.pid)
            if process.is_running():
                process.terminate()
            server_processes2.append(process)
        except Exception:
            pass
    server_processes = server_processes2
    index = 0
    while index < 100:  # wait max 10 s to exit
        exist_alive = False
        for process in server_processes:
            if process.is_running():
                exist_alive = True
                break
        if not exist_alive:
            return True
        time.sleep(0.1)
        index += 1

    for process in server_processes:
        if process.is_running():
            try:
                process.kill()
            except Exception:
                pass
    return False


g_redis_server_running = True
g_redis_with_ssl = False


def stop_redis_server():
    """stop redis server"""
    # SIGNAL 15
    cmd = f"pid=`ps aux | grep 'redis-server' | grep :{g_redis_server_port}"
    cmd += " | grep -v \"grep\" |awk '{print $2}'` && "
    cmd += "for id in $pid; do kill -15 $id && echo \"killed $id\"; done"
    subprocess.call(['bash', '-c', cmd])
    time.sleep(0.5)
    # SIGNAL 9
    cmd = f"pid=`ps aux | grep 'redis-server' | grep :{g_redis_server_port}"
    cmd += " | grep -v \"grep\" |awk '{print $2}'` && "
    cmd += "for id in $pid; do kill -9 $id && echo \"killed $id\"; done"
    subprocess.call(['bash', '-c', cmd])
    print(f"stop redis server {g_redis_server_port}")

    global g_redis_server_running
    g_redis_server_running = False


def start_redis_server():
    cmd = f"redis-server --port {g_redis_server_port} --save \"\" &"
    subprocess.call(['bash', '-c', cmd])
    time.sleep(0.5)
    print(f"start redis server {g_redis_server_port}")

    global g_redis_server_running, g_redis_with_ssl
    g_redis_server_running = True
    g_redis_with_ssl = False


def restart_redis_server():
    """restart redis server"""
    stop_redis_server()
    start_redis_server()


def start_redis_with_ssl():
    """start redis with ssl"""
    stop_redis_server()
    server_crt, server_key, _, _, ca_crt = get_default_redis_ssl_config()
    cmd = f"redis-server --port 0 --tls-port {g_redis_server_port} --tls-cert-file {server_crt} " \
          f"--tls-key-file {server_key} --tls-ca-cert-file {ca_crt} --save \"\" &"
    print(cmd)
    subprocess.call(['bash', '-c', cmd])
    time.sleep(0.5)
    print(f"start redis server {g_redis_server_port}")

    global g_redis_server_running, g_redis_with_ssl
    g_redis_server_running = True
    g_redis_with_ssl = True


def recover_redis_server():
    """recover redis server"""
    if g_redis_server_running and not g_redis_with_ssl:
        return
    stop_redis_server()
    start_redis_server()


def start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size, enable_ssl=None):
    """start fl job expect success"""
    for i in range(10):  # 0.5*10=5s
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size,
                                                           enable_ssl=enable_ssl)
        if client_feature_map is None:
            if isinstance(fl_job_rsp, str) and fl_job_rsp != server_safemode_rsp:
                raise RuntimeError(f"Failed to post startFLJob: {fl_job_rsp}")
            if isinstance(fl_job_rsp, ResponseFLJob.ResponseFLJob) and \
                    fl_job_rsp.Retcode() != ResponseCode.ResponseCode.SucNotReady:
                raise RuntimeError(
                    f"Failed to post startFLJob: {fl_job_rsp.Retcode()} {fl_job_rsp.Reason().decode()}")
        else:
            return client_feature_map, fl_job_rsp
        time.sleep(0.5)
    if isinstance(fl_job_rsp, str):
        raise RuntimeError(f"Failed to post startFLJob: {fl_job_rsp}")
    raise RuntimeError(f"Failed to post startFLJob: {fl_job_rsp.Retcode()} {fl_job_rsp.Reason().decode()}")


def update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map, upload_loss=0.0,
                                enable_ssl=None):
    """update model expect success"""
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id, iteration, update_feature_map,
                                                 upload_loss=upload_loss, enable_ssl=enable_ssl)
    if result is None:
        if isinstance(update_model_rsp, str):
            raise RuntimeError(f"Failed to post updateModel: {update_model_rsp}")
        raise RuntimeError(
            f"Failed to post updateModel: {update_model_rsp.Retcode()} {update_model_rsp.Reason().decode()}")
    return result, update_model_rsp


def get_model_expect_success(http_server_address, fl_name, iteration, enable_ssl=None):
    """get model expect success"""
    for i in range(60):  # 0.5*60=30s
        client_feature_map, get_model_rsp = post_get_model(http_server_address, fl_name, iteration,
                                                           enable_ssl=enable_ssl)
        if client_feature_map is None:
            if isinstance(get_model_rsp, str) and get_model_rsp != server_safemode_rsp:
                raise RuntimeError(f"Failed to post getModel: {get_model_rsp}")
            if isinstance(get_model_rsp, ResponseGetModel.ResponseGetModel) and \
                    get_model_rsp.Retcode() != ResponseCode.ResponseCode.SucNotReady:
                raise RuntimeError(
                    f"Failed to post getModel: {get_model_rsp.Retcode()} {get_model_rsp.Reason().decode()}")
        else:
            return client_feature_map, get_model_rsp
        time.sleep(0.5)
    if isinstance(get_model_rsp, str):
        raise RuntimeError(f"Failed to post getModel: {get_model_rsp}")
    raise RuntimeError(f"Failed to post getModel: {get_model_rsp.Retcode()} {get_model_rsp.Reason().decode()}")


def check_feature_map(expect_feature_map, result_feature_map):
    assert isinstance(expect_feature_map, dict)
    assert isinstance(result_feature_map, dict)
    assert len(expect_feature_map) == len(result_feature_map)
    for feature_name, expect in expect_feature_map.items():
        assert feature_name in result_feature_map
        result = result_feature_map[feature_name]
        assert isinstance(result, np.ndarray)
        if not (np.abs(result.reshape(-1) - expect.reshape(-1)) <= 0.0001).all():
            raise RuntimeError(f"feature: {feature_name}, expect: {expect.reshape(-1)}, result: {result.reshape(-1)}")


def post_scheduler_msg(http_address, request_url, post_data):
    try:
        resp = requests.post(f"http://{http_address}/{request_url}", data=post_data)
        return json.loads(resp.text)
    except Exception as e:
        return e


def post_scheduler_state_msg(scheduler_http_address):
    return post_scheduler_msg(scheduler_http_address, "state", None)


def post_scheduler_new_instance_msg(scheduler_http_address, post_msg):
    post_data = json.dumps(post_msg)
    return post_scheduler_msg(scheduler_http_address, "newInstance", post_data)


def post_scheduler_query_instance_msg(scheduler_http_address):
    return post_scheduler_msg(scheduler_http_address, "queryInstance", None)


def post_scheduler_disable_msg(scheduler_http_address):
    return post_scheduler_msg(scheduler_http_address, "disableFLS", None)


def post_scheduler_enable_msg(scheduler_http_address):
    return post_scheduler_msg(scheduler_http_address, "enableFLS", None)


def read_metrics(metrics_file="metrics.json"):
    if not os.path.exists(metrics_file):
        raise RuntimeError(f"Cannot find metrics file {metrics_file}")
    with open(metrics_file, "r") as fp:
        metrics_lines = fp.readlines()
    metrics = []
    for line in metrics_lines:
        metrics.append(json.loads(line))
    return metrics
