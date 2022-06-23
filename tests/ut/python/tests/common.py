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
import os
import signal
import logging
import time
import yaml
import socket
import psutil
import subprocess
import json
import numpy as np
import traceback
from multiprocessing import Process, Pipe
from functools import wraps
import requests

from mindspore_federated import log as logger
from mindspore_federated import FLServerJob, FeatureMap, FlSchedulerJob, Callback
from common_client import post_start_fl_job, post_get_model, post_update_model
from common_client import server_safemode_rsp, server_disabled_finished_rsp
from common_client import ResponseCode, ResponseFLJob, ResponseGetModel, ResponseUpdateModel


def fl_test(func):
    def clean_temp_files():
        cwd_dir = os.getcwd()
        temp_dir = os.path.join(cwd_dir, "temp")
        os.system(f"rm -rf {temp_dir}")
        fl_ckpt_dir = os.path.join(cwd_dir, "fl_ckpt")
        os.system(f"rm -rf {fl_ckpt_dir}")
        os.system(f"rm -f metrics.json")
        os.system(f"rm -f event.txt")

    @wraps(func)
    def wrap_test(*args, **kwargs):
        try:
            clean_temp_files()
            try:
                os.mkdir("temp")
            except FileExistsError:
                pass
            try:
                os.mkdir("fl_ckpt")
            except FileExistsError:
                pass
            # start_redis_server()
            func(*args, **kwargs)
        except Exception:
            logger.error("FL test catch exception")
            temp_dir = os.path.join(os.getcwd(), "temp")
            os.system(f"ls -l {temp_dir}/*.yaml && cat {temp_dir}/*.yaml")
            raise
        finally:
            logger.info("Fl test begin to clear")
            global g_server_processes
            stop_processes(g_server_processes)
            # stop_redis_server()
            clean_temp_files()
            logger.info("Fl test end clear")

    return wrap_test


g_server_processes = []
g_redis_server_port = int(os.environ["REDIS_SERVER_PORT"])
g_redis_server_address = f"127.0.0.1:{g_redis_server_port}"

g_fl_name_idx = 1
fl_training_mode = "FEDERATED_LEARNING"
fl_hybrid_mode = "HYBRID_TRAINING"


def fl_name_with_idx(fl_name):
    global g_fl_name_idx
    new_fl_name = f"fl_name_{g_fl_name_idx}"
    g_fl_name_idx += 1
    return new_fl_name


def make_yaml_config(fl_name, update_configs, output_yaml_file,
                     server_mode=None, distributed_cache_address=None, fl_iteration_num=None,
                     start_fl_job_threshold=None, update_model_ratio=None,
                     start_fl_job_time_window=None, update_model_time_window=None, global_iteration_time_window=None):
    with open("default_yaml_config.yaml") as fp:
        yaml_file_content = fp.read()
    yaml_configs = yaml.load(yaml_file_content, yaml.Loader)
    update_configs["fl_name"] = fl_name

    def set_when_not_none(dst, key, val):
        if val is not None:
            dst[key] = val

    set_when_not_none(update_configs, "server_mode", server_mode)
    set_when_not_none(update_configs, "fl_iteration_num", fl_iteration_num)
    if "round" not in update_configs:
        update_configs["round"] = {}
    round_configs = update_configs["round"]
    set_when_not_none(round_configs, "start_fl_job_threshold", start_fl_job_threshold)
    set_when_not_none(round_configs, "update_model_ratio", update_model_ratio)
    set_when_not_none(round_configs, "start_fl_job_time_window", start_fl_job_time_window)
    set_when_not_none(round_configs, "update_model_time_window", update_model_time_window)
    set_when_not_none(round_configs, "global_iteration_time_window", global_iteration_time_window)

    if "distributed_cache" not in update_configs:
        update_configs["distributed_cache"] = {}
    cache_configs = update_configs["distributed_cache"]
    set_when_not_none(cache_configs, "address", distributed_cache_address)

    def update_one_dict(dst_dict, src_dict):
        for key, val in src_dict.items():
            if key not in dst_dict:
                dst_dict[key] = val
            elif isinstance(val, dict):
                update_one_dict(dst_dict[key], val)
            else:
                dst_dict[key] = val

    update_one_dict(yaml_configs, update_configs)
    new_yaml_content = yaml.dump(yaml_configs, Dumper=yaml.Dumper)
    with open(output_yaml_file, "w") as fp:
        fp.write(new_yaml_content)


def start_fl_server(feature_map, yaml_config, http_server_address, tcp_server_ip="127.0.0.1",
                    checkpoint_dir="./fl_ckpt/"):
    print("new server process", flush=True)
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    send_pipe, recv_pipe = Pipe()

    class FlCallback(Callback):
        def __init__(self):
            super(FlCallback, self).__init__()

        def after_started(self):
            send_pipe.send("Success")

    callback = FlCallback()

    def server_process_fun():
        try:
            server_job = FLServerJob(yaml_config, http_server_address, tcp_server_ip, checkpoint_dir)
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
    while index < 100:  # wait max 10 s
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
    assert index < 100
    return server_process


def start_fl_scheduler(yaml_config, scheduler_http_address):
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
    except:
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


def stop_redis_server():
    cmd = f"pid=`ps aux | grep 'redis-server' | grep :{g_redis_server_port}"
    cmd += " | grep -v \"grep\" |awk '{print $2}'` && "
    cmd += "for id in $pid; do kill -9 $id && echo \"killed $id\"; done"
    subprocess.call(['bash', '-c', cmd])
    print(f"stop redis server {g_redis_server_port}")


def start_redis_server():
    stop_redis_server()
    cmd = f"redis-server --port {g_redis_server_port} &"
    subprocess.call(['bash', '-c', cmd])
    print(f"start redis server {g_redis_server_port}")


def start_fl_job_expect_success(http_server_address, fl_name, fl_id, data_size):
    for i in range(10):  # 0.5*10=5s
        client_feature_map, fl_job_rsp = post_start_fl_job(http_server_address, fl_name, fl_id, data_size)
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


def update_model_expect_success(http_server_address, fl_name, fl_id, iteration, update_feature_map, upload_loss=0.0):
    result, update_model_rsp = post_update_model(http_server_address, fl_name, fl_id, iteration, update_feature_map,
                                                 upload_loss=upload_loss)
    if result is None:
        if isinstance(update_model_rsp, str):
            raise RuntimeError(f"Failed to post updateModel: {update_model_rsp}")
        raise RuntimeError(
            f"Failed to post updateModel: {update_model_rsp.Retcode()} {update_model_rsp.Reason().decode()}")
    return result, update_model_rsp


def get_model_expect_success(http_server_address, fl_name, iteration):
    # get model
    for i in range(10):  # 0.5*10=5s
        client_feature_map, get_model_rsp = post_get_model(http_server_address, fl_name, iteration)
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


def post_scheduler_stop_msg(scheduler_http_address):
    return post_scheduler_msg(scheduler_http_address, "stopFLS", None)


def read_metrics(metrics_file="metrics.json"):
    if not os.path.exists(metrics_file):
        raise RuntimeError(f"Cannot find metrics file {metrics_file}")
    with open(metrics_file, "r") as fp:
        metrics_lines = fp.readlines()
    metrics = []
    for line in metrics_lines:
        metrics.append(json.loads(line))
    return metrics
