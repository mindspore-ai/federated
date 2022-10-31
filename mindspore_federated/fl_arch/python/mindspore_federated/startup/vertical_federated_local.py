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
"""Interface for start up vertical federated communicator"""
from collections import OrderedDict

from mindspore_federated._mindspore_federated import VerticalFederated_
from mindspore_federated.common import tensor_utils, data_join_utils
from mindspore_federated.data_join.context import _WorkerRegister
from .ssl_config import init_ssl_config, SSLConfig
from .server_config import ServerConfig, init_server_config


class VerticalFederatedCommunicator:
    """
    Define the vertical federated communicator.

    Args:
        http_server_config (ServerConfig): Configuration of local http server.
        remote_server_config (ServerConfig): Configuration of remote http server.
    Examples:
        >>> from mindspore_federated import VerticalFederatedCommunicator, ServerConfig
        >>> http_server_config = ServerConfig(server_name='server', server_address="127.0.0.1:1086")
        >>> remote_server_config = ServerConfig(server_name='client', server_address="127.0.0.1:1087")
        >>> vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
        ...                                                       remote_server_config=remote_server_config)
        >>> vertical_communicator.launch()
        >>> vertical_communicator.data_join_wait_for_start()
    """

    def __init__(self, http_server_config: ServerConfig, remote_server_config: ServerConfig,
                 ssl_config=None):
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
        self._http_server_config = http_server_config
        self._remote_server_config = remote_server_config
        self._ssl_config = ssl_config
        init_ssl_config(self._ssl_config)
        init_server_config(self._http_server_config, self._remote_server_config)

    def launch(self):
        """
        Start vertical federated learning communicator.
        """
        VerticalFederated_.start_vertical_communicator()

    def send_tensors(self, target_server_name: str, tensor_dict: OrderedDict):
        """
        Send distributed training sensor data.

        Args:
            target_server_name (str): Specifies the name of the remote server.
            tensor_dict (OrderedDict): The dict of Tensors to be sent.
        """
        tensor_list_item_py = tensor_utils.tensor_dict_to_tensor_list_pybind_obj(tensor_dict)
        return VerticalFederated_.send_tensor_list(target_server_name, tensor_list_item_py)

    def send_register(self, target_server_name: str, worker_register: _WorkerRegister):
        r"""
        Send worker registration message.

        Args:
            target_server_name (str): Specifies the name of the remote server.
            worker_register (_\WorkerRegister): The worker registration information to be sent.
        """
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
