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
from mindspore_federated._mindspore_federated import VerticalFederated_, VFLContext, TensorListItem_
from .ssl_config import init_ssl_config, SSLConfig
from ..common import check_type


class VFLTrainer:
    """
    Define the vertical trainer communicator
    """

    def __init__(self, http_server_address, remote_http_address, ssl_config=None):
        check_type.check_str("http_server_address", http_server_address)

        if ssl_config is not None and not isinstance(ssl_config, SSLConfig):
            raise RuntimeError(
                f"Parameter 'ssl_config' should be None or instance of SSLConfig, but got {type(ssl_config)}")

        ctx = VFLContext.get_instance()
        ctx.set_http_server_address(http_server_address)
        ctx.set_remote_server_address(remote_http_address)
        init_ssl_config(ssl_config)

    def start_communicator(self):
        VerticalFederated_.start_vertical_communicator()

    def send(self, tensor_list_item: TensorListItem_):
        return VerticalFederated_.send(tensor_list_item)

    def receive(self):
        return VerticalFederated_.receive()
