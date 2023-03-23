# pylint: disable=missing-docstring
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

"""Interface for start up single core servable"""
import os.path
import numpy as np
from mindspore_federated._mindspore_federated import Federated_, FLContext, FeatureItem_
from ..common import _fl_context
from .feature_map import FeatureMap
from .. import log as logger
from .ssl_config import init_ssl_config, SSLConfig
from ..common import check_type
from .yaml_config import load_yaml_config


def load_ms_checkpoint(checkpoint_file_path):
    """
    load ms checkpoint
    """
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
    """
    load mindir
    """
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
    """
    define callback of fl server job
    """

    def __init__(self):
        pass

    def after_started(self):
        """
        Callback after the server is successfully started.
        """

    def before_stopped(self):
        """
        Callback after the server is stopped.
        """

    def on_iteration_end(self, context):
        """
        Callback at the end of one iteration.

        Args:
            context (CallbackContext): Context of the iteration.
        """


class FLServerJob:
    """
    Define Federated Learning cloud-side tasks.

    Args:
        yaml_config (str): The yaml file path. More detail see `federated_server_yaml <https://gitee.com/mindspore/federated/blob/master/docs/api/api_python_en/horizontal/federated_server_yaml.md>`_.
        http_server_address (str): The http server address used for communicating.
        tcp_server_ip (str): The tcp server ip used for communicating. Default: "127.0.0.1".
        checkpoint_dir (str): Path of checkpoint. Default: "./fl_ckpt/".
        ssl_config (Union(None, SSLConfig)) : Config of ssl. Default: None.

    Examples:
        >>> job = FLServerJob(yaml_config=yaml_config, http_server_address=http_server_address,
        ...                   tcp_server_ip=tcp_server_ip, checkpoint_dir=checkpoint_dir)
        >>> job.run()
    """

    def __init__(self, yaml_config, http_server_address, tcp_server_ip="127.0.0.1",
                 checkpoint_dir="./fl_ckpt/", ssl_config=None):
        check_type.check_str("yaml_config", yaml_config)
        check_type.check_str("http_server_address", http_server_address)
        check_type.check_str("tcp_server_ip", tcp_server_ip)
        check_type.check_str("checkpoint_dir", checkpoint_dir)

        if ssl_config is not None and not isinstance(ssl_config, SSLConfig):
            raise RuntimeError(
                f"Parameter 'ssl_config' should be None or instance of SSLConfig, but got {type(ssl_config)}")

        ctx = FLContext.get_instance()
        ctx.set_http_server_address(http_server_address)
        ctx.set_tcp_server_ip(tcp_server_ip)
        ctx.set_checkpoint_dir(checkpoint_dir)
        init_ssl_config(ssl_config)
        load_yaml_config(yaml_config, _fl_context.ROLE_OF_SERVER)
        self.checkpoint_dir = checkpoint_dir
        self.fl_name = ctx.fl_name()
        self.aggregation_type = ctx.aggregation_type()
        self.callback = None

    def run(self, feature_map=None, callback=None):
        """
        Run fl server job.

        Args:
            feature_map (Union(dict, FeatureMap, str)): Feature map. Default: None.
            callback (Union(None, Callback)): Callback function. Default: None.
        """
        if callback is not None and not isinstance(callback, Callback):
            raise RuntimeError("Parameter 'callback' is expected to be instance of Callback when it's not None, but"
                               f" got {type(callback)}.")
        self.callback = callback
        recovery_ckpt_files = self._get_current_recovery_ckpt_files()
        feature_map = self._load_feature_map(feature_map, recovery_ckpt_files)
        recovery_iteration = self._get_current_recovery_iteration(recovery_ckpt_files)
        feature_list_cxx = []
        for _, feature in feature_map.feature_map().items():
            feature_cxx = FeatureItem_(feature.feature_name, feature.data, feature.shape, "fp32",
                                       feature.require_aggr)
            feature_list_cxx.append(feature_cxx)
        if self.aggregation_type == _fl_context.SCAFFOLD:
            for _, feature in feature_map.feature_map().items():
                feature_cxx = FeatureItem_("control." + feature.feature_name, np.zeros_like(feature.data),
                                           feature.shape, "fp32", feature.require_aggr)
                feature_list_cxx.append(feature_cxx)
        Federated_.start_federated_server(feature_list_cxx, recovery_iteration, self.after_started_callback,
                                          self.before_stopped_callback, self.on_iteration_end_callback)

    def after_started_callback(self):
        logger.info("after started callback")
        if self.callback is not None:
            try:
                self.callback.after_started()
            except RuntimeError as e:
                logger.warning(f"Catch exception when invoke callback after started: {str(e)}.")

    def before_stopped_callback(self):
        logger.info("before stopped callback")
        if self.callback is not None:
            try:
                self.callback.before_stopped()
            except RuntimeError as e:
                logger.warning(f"Catch exception when invoke callback before stopped: {str(e)}.")

    def on_iteration_end_callback(self, feature_list, fl_name, instance_name, iteration_num,
                                  iteration_valid, iteration_reason):
        logger.info("on iteration end callback.")
        feature_map = {}
        checkpoint_file = ""
        if os.path.exists(self.checkpoint_dir):
            feature_map = FeatureMap()
            for feature in feature_list:
                feature_map.add_feature(feature.feature_name, feature.data, feature.require_aggr)
            checkpoint_file = self._save_feature_map(feature_map, iteration_num)
        if self.callback is not None:
            try:
                context = CallbackContext(feature_map, checkpoint_file, fl_name, instance_name,
                                          iteration_num, iteration_valid, iteration_reason)
                self.callback.on_iteration_end(context)
            except RuntimeError as e:
                logger.warning(f"Catch exception when invoke callback on iteration end: {str(e)}.")

    def _save_feature_map(self, feature_map, iteration_num):
        """
        save feature map.
        """
        recovery_ckpt_files = self._get_current_recovery_ckpt_files()
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{self.fl_name}_recovery_iteration_{iteration_num}_{timestamp}.ckpt"
        new_ckpt_file_path = os.path.join(self.checkpoint_dir, file_name)
        save_ms_checkpoint(new_ckpt_file_path, feature_map)
        if len(recovery_ckpt_files) >= 3:
            for _, _, file in recovery_ckpt_files[2:]:
                os.remove(file)
        return new_ckpt_file_path

    def _load_feature_map(self, feature_map, recovery_ckpt_files):
        """
        load feature map.
        """
        if isinstance(feature_map, dict):
            new_feature_map = FeatureMap()
            for feature_name, val in feature_map.items():
                new_feature_map.add_feature(feature_name, val, require_aggr=True)
            feature_map = new_feature_map

        # load checkpoint file in self.checkpoint_dir
        if recovery_ckpt_files:
            feature_map_ckpt = None
            for _, _, ckpt_file in recovery_ckpt_files:
                try:
                    feature_map_ckpt = load_ms_checkpoint(ckpt_file)
                    logger.info(f"Load recovery checkpoint file {ckpt_file} successfully.")
                    break
                except ValueError as e:
                    logger.warning(f"Failed to load recovery checkpoint file {ckpt_file}: {str(e)}.")
                    continue
            if feature_map_ckpt is not None:
                if not isinstance(feature_map, FeatureMap):
                    return feature_map_ckpt
                feature_map_dict = feature_map.feature_map()
                for key, val in feature_map_ckpt.feature_map().items():
                    if key in feature_map_dict:
                        val.require_aggr = feature_map_dict[key].require_aggr
                return feature_map_ckpt

        if isinstance(feature_map, FeatureMap):
            return feature_map
        if isinstance(feature_map, str):
            if feature_map.endswith(".ckpt"):
                return load_ms_checkpoint(feature_map)
            if feature_map.endswith(".mindir"):
                return load_mindir(feature_map)
            raise RuntimeError(f"The value of parameter 'feature_map' is expected to be checkpoint file path, "
                               f"ends with '.ckpt', or MindIR file path, ends with '.mindir', "
                               f"when the type of parameter 'feature_map' is str.")
        raise RuntimeError(
            f"The parameter 'feature_map' is expected to be instance of dict(feature_name, feature_val), FeatureMap, "
            f"or a checkpoint or mindir file path, but got '{type(feature_map)}'.")

    def _get_current_recovery_ckpt_files(self):
        """
        get current recovery ckpt file.
        """
        # get checkpoint files from the latest to the next new in self.checkpoint_dir: {checkpoint_dir}/
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
        recovery_ckpt_files.sort(key=lambda elem: elem[0], reverse=True)
        logger.info(f"Recovery ckpt files is: {recovery_ckpt_files}.")
        return recovery_ckpt_files

    def _get_current_recovery_iteration(self, recovery_ckpt_files):
        """
        get current recovery iteration.
        """
        recovery_iteration = 1
        if not recovery_ckpt_files:
            return recovery_iteration
        for iteration_num, _, _ in recovery_ckpt_files:
            recovery_iteration = int(iteration_num) + 1
            break
        logger.info(f"Recovery iteration num is: {recovery_iteration}.")
        return recovery_iteration


class FlSchedulerJob:
    """
    Define federated scheduler job.

    Args:
        yaml_config (str): The yaml file path. More detail see `federated_server_yaml <https://gitee.com/mindspore/federated/blob/master/docs/api/api_python_en/horizontal/federated_server_yaml.md>`_.
        manage_address (str): The management address.
        ssl_config (Union(None, SSLConfig)): Config of ssl. Default: None.

    Examples:
        >>> job = FlSchedulerJob(yaml_config=yaml_config, manage_address=scheduler_manage_address)
        >>> job.run()
    """

    def __init__(self, yaml_config, manage_address, ssl_config=None):
        check_type.check_str("yaml_config", yaml_config)
        check_type.check_str("manage_address", manage_address)

        if ssl_config is not None and not isinstance(ssl_config, SSLConfig):
            raise RuntimeError(
                f"Parameter 'ssl_config' should be None or instance of SSLConfig, but got {type(ssl_config)}")

        ctx = FLContext.get_instance()
        ctx.set_scheduler_manage_address(manage_address)
        init_ssl_config(ssl_config)
        load_yaml_config(yaml_config, _fl_context.ROLE_OF_SCHEDULER)

    def run(self):
        """
        Run scheduler job.
        """
        Federated_.start_federated_scheduler()
