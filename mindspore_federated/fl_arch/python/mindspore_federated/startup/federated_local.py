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
"""Interface for start up single core servable"""
import os.path
import numpy as np
from ..common import _fl_context
from .feature_map import FeatureMap
from .. import log as logger
from .ssl_config import init_ssl_config
from .yaml_config import load_yaml_config
from mindspore_federated._mindspore_federated import Federated_, FLContext, FeatureItem_


def load_ms_checkpoint(checkpoint_file_path):
    logger.info(f"load checkpoint file: {checkpoint_file_path}")
    from mindspore import load_checkpoint
    param_dict = load_checkpoint(checkpoint_file_path)

    feature_map = FeatureMap()
    for param_name, param_value in param_dict.items():
        weight_np = param_value.asnumpy()
        if weight_np.dtype != np.float32:
            logger.info(f"Skip weight {param_name}, type is {weight_np.dtype}")
            continue
        feature_map.add_feature(param_name, weight_np, True)
        logger.info(f"Weight name: {param_name}, shape: {list(weight_np.shape)}, dtype: {weight_np.dtype}")
    return feature_map


def save_ms_checkpoint(checkpoint_file_path, feature_map):
    logger.info(f"save checkpoint file: {checkpoint_file_path}")
    from mindspore import save_checkpoint, Tensor
    if not isinstance(feature_map, FeatureMap):
        raise RuntimeError(
            f"Parameter 'feature_map' is expected to be instance of FeatureMap, but got {type(feature_map)}")
    params = []
    for feature_name, feature in feature_map.feature_map().items():
        params.append({"name": feature_name, "data": Tensor(feature.data)})
    save_checkpoint(params, checkpoint_file_path)


def load_mindir(mindir_file_path):
    logger.info(f"load MindIR file: {mindir_file_path}")
    from mindspore import load, nn
    graph = load(mindir_file_path)
    graph_cell = nn.GraphCell(graph)
    feature_map = FeatureMap()

    for _, param in graph_cell.parameters_and_names():
        param_name = param.name
        weight_np = param.data.asnumpy()
        if weight_np.dtype != np.float32:
            logger.info(f"Skip weight {param_name}, type is {weight_np.dtype}")
            continue
        feature_map.add_feature(param_name, weight_np, True)
        logger.info(f"Weight name: {param_name}, shape: {list(weight_np.shape)}, dtype: {weight_np.dtype}")
    return feature_map


class CallbackContext:
    def __init__(self, feature_map, checkpoint_file, fl_name, instance_name,
                 iteration_num, iteration_valid, iteration_result):
        self.feature_map = feature_map
        self.checkpoint_file = checkpoint_file
        self.fl_name = fl_name
        self.instance_name = instance_name
        self.iteration_num = iteration_num
        self.iteration_valid = iteration_valid
        self.iteration_result = iteration_result


class Callback:
    def __init__(self):
        pass

    def after_started(self):
        """
        Callback after the server is successfully started.
        """
        pass

    def before_stopped(self):
        """
        Callback after the server is stopped.
        """
        pass

    def on_iteration_end(self, context):
        """
        Callback at the end of one iteration.

        Args:
            context (CallbackContext): Context of the iteration.
        """
        pass


class SSLConfig:
    def __init__(self, server_password, client_password):
        self.server_password = server_password
        self.client_password = client_password


def _check_str(arg_name, str_val):
    """Check whether the input parameters are reasonable str input"""
    if not isinstance(str_val, str):
        raise RuntimeError(f"Parameter '{arg_name}' should be str, but actually {type(str_val)}")
    if not str_val:
        raise RuntimeError(f"Parameter '{arg_name}' should not be empty str")


class FLServerJob:
    def __init__(self, yaml_config, http_server_address, tcp_server_ip="127.0.0.1",
                 checkpoint_dir="./fl_ckpt/", ssl_config=None):
        _check_str("yaml_config", yaml_config)
        _check_str("http_server_address", http_server_address)
        _check_str("tcp_server_ip", tcp_server_ip)
        _check_str("checkpoint_dir", checkpoint_dir)

        if ssl_config is not None and not isinstance(ssl_config, SSLConfig):
            raise RuntimeError(
                f"Parameter 'ssl_config' should be None or instance of SSLConfig, but actually {type(ssl_config)}")

        ctx = FLContext.get_instance()
        ctx.set_http_server_address(http_server_address)
        ctx.set_tcp_server_ip(tcp_server_ip)
        ctx.set_checkpoint_dir(checkpoint_dir)
        enable_ssl = init_ssl_config(ssl_config)
        load_yaml_config(yaml_config, _fl_context.RoleOfServer, enable_ssl)

        self.checkpoint_dir = checkpoint_dir
        self.fl_name = ctx.fl_name()
        self.callback = None

    def run(self, feature_map=None, callback=None):
        if callback is not None and not isinstance(callback, Callback):
            raise RuntimeError("Parameter 'callback' is expected to be instance of Callback when it's not None, but"
                               f" got {type(callback)}")
        self.callback = callback
        feature_map = self._load_feature_map(feature_map)
        feature_list_cxx = []
        for _, feature in feature_map.feature_map().items():
            feature_cxx = FeatureItem_(feature.feature_name, feature.data, feature.shape, "fp32",
                                       feature.requires_aggr)
            feature_list_cxx.append(feature_cxx)
        Federated_.start_federated_server(feature_list_cxx, self.after_started_callback,
                                          self.before_stopped_callback, self.on_iteration_end_callback)

    def after_started_callback(self):
        logger.info("after started callback")
        if self.callback is not None:
            try:
                self.callback.after_started()
            except Exception as e:
                logger.warning(f"Catch exception when invoke callback after started: {str(e)}")

    def before_stopped_callback(self):
        logger.info("before stopped callback")
        if self.callback is not None:
            try:
                self.callback.before_stopped()
            except Exception as e:
                logger.warning(f"Catch exception when invoke callback before stopped: {str(e)}")

    def on_iteration_end_callback(self, feature_list, fl_name, instance_name, iteration_num,
                                  iteration_valid, iteration_reason):
        logger.info("on iteration end callback")
        if self.callback is not None:
            try:
                feature_map = FeatureMap()
                for feature in feature_list:
                    feature_map.add_feature(feature.feature_name, feature.data, feature.requires_aggr)
                checkpoint_file = self._save_feature_map(feature_map, iteration_num)
                context = CallbackContext(feature_map, checkpoint_file, fl_name, instance_name,
                                          iteration_num, iteration_valid, iteration_reason)
                self.callback.on_iteration_end(context)
            except Exception as e:
                logger.warning(f"Catch exception when invoke callback on iteration end: {str(e)}")

    def _save_feature_map(self, feature_map, iteration_num):
        recovery_ckpt_file = self._get_current_recovery_ckpt_file()
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{self.fl_name}_recovery_iteration_{iteration_num}_{timestamp}.ckpt"
        new_ckpt_file_path = os.path.join(self.checkpoint_dir, file_name)
        save_ms_checkpoint(new_ckpt_file_path, feature_map)
        if len(recovery_ckpt_file) >= 3:
            for _, _, file in recovery_ckpt_file[:-2]:
                os.remove(file)
        return new_ckpt_file_path

    def _load_feature_map(self, feature_map):
        if isinstance(feature_map, dict):
            new_feature_map = FeatureMap()
            for feature_name, val in feature_map.items():
                new_feature_map.add_feature(feature_name, val, requires_aggr=True)
            feature_map = new_feature_map

        # load checkpoint file in self.checkpoint_dir
        recovery_ckpt_file = self._get_current_recovery_ckpt_file()
        if recovery_ckpt_file:
            latest_ckpt_file = recovery_ckpt_file[-1][2]
            feature_map_ckpt = load_ms_checkpoint(latest_ckpt_file)
            if not isinstance(feature_map, FeatureMap):
                return feature_map_ckpt
            feature_map_dict = feature_map.feature_map()
            for key, val in feature_map_ckpt.feature_map().items():
                if key in feature_map_dict:
                    val.requires_aggr = feature_map_dict[key].requires_aggr
            return feature_map_ckpt

        if isinstance(feature_map, FeatureMap):
            return feature_map
        if isinstance(feature_map, str):
            if feature_map.endswith(".ckpt"):
                return load_ms_checkpoint(feature_map)
            elif feature_map.endswith(".mindir"):
                return load_mindir(feature_map)
            raise RuntimeError(f"The value of parameter 'feature_map' is expected to be checkpoint file path, "
                               f"ends with '.ckpt', or MindIR file path, ends with '.mindir', "
                               f"when the type of parameter 'feature_map' is str.")
        raise RuntimeError(
            f"The parameter 'feature_map' is expected to be instance of dict(feature_name, feature_val), FeatureMap, "
            f"or a checkpoint or mindir file path, but got '{type(feature_map)}'")

    def _get_current_recovery_ckpt_file(self):
        # get checkpoint file in self.checkpoint_dir: {checkpoint_dir}/
        # checkpoint file: {fl_name}_recovery_iteration_xxx_20220601_164030.ckpt
        if not os.path.exists(self.checkpoint_dir) or not os.path.isdir(self.checkpoint_dir):
            return None
        prefix = f"{self.fl_name}_recovery_iteration_"
        postfix = ".ckpt"
        filelist = os.listdir(self.checkpoint_dir)
        recovery_ckpt_files = []
        for file in filelist:
            file_path = os.path.join(self.checkpoint_dir, file)
            if not os.path.isfile(file_path):
                continue
            if file[:len(prefix)] == prefix and file[-len(postfix):] == postfix:
                strs = file[len(prefix):-len(postfix)].split("_")
                if len(strs) != 3:
                    continue
                iteration_num = int(strs[0])
                timestamp = strs[1] + strs[2]
                recovery_ckpt_files.append((iteration_num, timestamp, file_path))
        recovery_ckpt_files.sort(key=lambda elem: elem[1])
        return recovery_ckpt_files


class FlSchedulerJob:
    def __init__(self, yaml_config, manage_address):
        _check_str("yaml_config", yaml_config)
        _check_str("manage_address", manage_address)

        ctx = FLContext.get_instance()
        ctx.set_scheduler_manage_address(manage_address)
        load_yaml_config(yaml_config, _fl_context.RoleOfScheduler, enable_ssl=False)

    def run(self):
        Federated_.start_federated_scheduler()
