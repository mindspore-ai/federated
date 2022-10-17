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
from mindspore_federated._mindspore_federated import VerticalFederated_, TensorListItem_, WorkerRegisterItemPy_
from .ssl_config import init_ssl_config, SSLConfig
from .server_config import ServerConfig, init_server_config


class VerticalFederatedCommunicator:
    """
    Define the vertical communicator
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
        VerticalFederated_.start_vertical_communicator()

    def send_tensors(self, target_server_name: str, tensor_list_item_py: TensorListItem_):
        return VerticalFederated_.send_tensor_list(target_server_name, tensor_list_item_py)

    def send_register(self, target_server_name: str, worker_register_item_py: WorkerRegisterItemPy_):
        return VerticalFederated_.send_worker_register(target_server_name, worker_register_item_py)

    def receive(self, target_server_name: str):
        return VerticalFederated_.receive(target_server_name)

    def data_join_wait_for_start(self):
        return VerticalFederated_.data_join_wait_for_start()

    def http_server_config(self):
        return self._http_server_config

    def remote_server_config(self):
        return self._remote_server_config
