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
"""SSL config for start up FL"""
from mindspore_federated._mindspore_federated import FLContext
from ..common import check_type


class SSLConfig:
    def __init__(self, server_password, client_password):
        check_type.check_str("server_password", server_password)
        check_type.check_str("client_password", client_password)
        self.server_password = server_password
        self.client_password = client_password


def init_ssl_config(ssl_config):
    ctx = FLContext.get_instance()
    if ssl_config is not None:
        if not isinstance(ssl_config, SSLConfig):
            raise RuntimeError(f"Parameter 'ssl_config' should be instance of SSLConfig, but got {type(ssl_config)}")
        ctx.set_server_password(ssl_config.server_password)
        ctx.set_client_password(ssl_config.client_password)
