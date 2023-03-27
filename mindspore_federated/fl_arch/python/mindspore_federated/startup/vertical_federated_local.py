# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Interface for start up vertical federated communicator"""
from mindspore_federated._mindspore_federated import VerticalFederated_
from mindspore_federated.common import tensor_utils, data_join_utils
from mindspore_federated.data_join.context import _WorkerRegister

from .server_config import ServerConfig, init_server_config
from .ssl_config import SSLConfig, init_vertical_ssl_config, init_vertical_enable_ssl


class VerticalFederatedCommunicator:
    """
    Define the vertical federated communicator.

    Args:
        http_server_config (ServerConfig): Configuration of local http server.
        remote_server_config (ServerConfig): Configuration of remote http server.
        enable_ssl (bool, optional): whether to enable ssl communication. Default: False.
        ssl_config (SSLConfig, optional): Configuration of ssl encryption. Default: None.
        compress_configs (dict, optional): Configuration of communication compression. Default: None.

    Examples:
        >>> from mindspore_federated import VerticalFederatedCommunicator, ServerConfig
        >>> http_server_config = ServerConfig(server_name='server', server_address="127.0.0.1:1086")
        >>> remote_server_config = ServerConfig(server_name='client', server_address="127.0.0.1:1087")
        >>> vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
        ...                                                       remote_server_config=remote_server_config)
        >>> vertical_communicator.launch()
        >>> vertical_communicator.data_join_wait_for_start()
    """

    def __init__(self, http_server_config, remote_server_config, enable_ssl=False,
                 ssl_config=None, compress_configs=None):
        if http_server_config is not None and not isinstance(http_server_config, ServerConfig):
            raise RuntimeError(
                f"Parameter 'http_server_config' should be instance of ServerConfig,"
                f"but got {type(http_server_config)}")
        if remote_server_config is not None and not isinstance(remote_server_config, ServerConfig) \
                and not isinstance(remote_server_config, list):
            raise RuntimeError(
                f"Parameter 'remote_server_address' should be instance of ServerConfig or list,"
                f"but got {type(remote_server_config)}")

        if ssl_config is not None and not isinstance(ssl_config, SSLConfig):
            raise RuntimeError(
                f"Parameter 'ssl_config' should be None or instance of SSLConfig, but got {type(ssl_config)}")
        if compress_configs is not None and not isinstance(compress_configs, dict):
            raise RuntimeError(
                f"Parameter 'compress_configs' should be None or instance of dict, but got {type(compress_configs)}")

        self._http_server_config = http_server_config
        self._remote_server_config = remote_server_config
        self._enable_ssl = enable_ssl
        self._ssl_config = ssl_config
        init_vertical_enable_ssl(self._enable_ssl)
        init_vertical_ssl_config(self._ssl_config)
        init_server_config(self._http_server_config, self._remote_server_config)
        self._compress_configs = compress_configs if compress_configs is not None else {}

    def launch(self):
        """
        Start vertical federated learning communicator.
        """
        VerticalFederated_.start_vertical_communicator()

    def send_tensors(self, target_server_name, tensor_dict):
        """
        Send distributed training sensor data.

        Args:
            target_server_name (str): Specifies the name of the remote server.
            tensor_dict (OrderedDict): The dict of Tensors to be sent.

        Examples:
            >>> from mindspore import Tensor
            >>> backbone_out = OrderedDict()
            >>> backbone_out["hidden_states"] = Tensor(np.random.random(size=(2, 2048, 1280)).astype(np.float16))
            >>> vertical_communicator.send_tensors("leader", backbone_out)
        """
        tensor_list_item_py = tensor_utils.tensor_dict_to_tensor_list_pybind_obj(
            ts_dict=tensor_dict, name="", compress_configs=self._compress_configs)
        return VerticalFederated_.send_tensor_list(target_server_name, tensor_list_item_py)

    def send_register(self, target_server_name: str, worker_register: _WorkerRegister):
        worker_register_item_py = data_join_utils.worker_register_to_pybind_obj(worker_register)
        worker_config_item_py = VerticalFederated_.send_worker_register(target_server_name, worker_register_item_py)
        return data_join_utils.pybind_obj_to_worker_config(worker_config_item_py)

    def receive(self, target_server_name: str):
        """
        Get the sensor data sent by the remote server.

        Args:
            target_server_name (str): Specifies the name of the remote server.
        """
        tensor_list_item_py = VerticalFederated_.receive(target_server_name)
        _, tensor_dict = tensor_utils.tensor_list_pybind_obj_to_tensor_dict(tensor_list_item_py)
        return tensor_dict

    def data_join_wait_for_start(self):
        """
        Block and wait for the registration information of the client worker.
        """
        return VerticalFederated_.data_join_wait_for_start()

    def http_server_config(self):
        """
        Returns the local server configuration.
        """
        return self._http_server_config

    def remote_server_config(self):
        """
        Returns the remote server configuration.
        """
        return self._remote_server_config
