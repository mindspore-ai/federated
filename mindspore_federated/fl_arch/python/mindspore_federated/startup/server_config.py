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
"""SSL config for start up FL"""
from typing import OrderedDict

from mindspore_federated._mindspore_federated import VFLContext

from ..common import check_type


class ServerConfig:
    """
    Define the vertical server configuration.

    Args:
        server_name (str): Name of server, such as "leader_server", user defined.
        server_address (str): Address of server, such as 127.0.0.1:1086, user defined.
    """
    def __init__(self, server_name, server_address):
        check_type.check_str("server_name", server_name)
        check_type.check_str("server_address", server_address)
        self.server_name = server_name
        self.server_address = server_address


def init_server_config(http_server_config, remote_server_config):
    """
    Initialize local server configuration and remote server configuration.

    Args:
        http_server_config (ServerConfig): Configuration of local http server.
        remote_server_config (ServerConfig): Configuration of remote http server.
    """
    ctx = VFLContext.get_instance()
    check_type.check_str("http_server_config.server_name", http_server_config.server_name)
    check_type.check_str("http_server_config.server_address", http_server_config.server_address)
    ctx.set_http_server_name(http_server_config.server_name)
    ctx.set_http_server_address(http_server_config.server_address)

    remote_server_dict = OrderedDict()
    if isinstance(remote_server_config, ServerConfig):
        check_type.check_str("remote_server_config.server_name", remote_server_config.server_name)
        check_type.check_str("remote_server_config.server_address", remote_server_config.server_address)
        remote_server_dict[remote_server_config.server_name] = remote_server_config.server_address

    elif isinstance(remote_server_config, list):
        for item in remote_server_config:
            check_type.check_str("remote_server_config.server_name", item.server_name)
            check_type.check_str("remote_server_config.server_address", item.server_address)
            remote_server_dict[item.server_name] = item.server_address
    ctx.set_remote_server_address(remote_server_dict)
