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
from mindspore_federated._mindspore_federated import FLContext, VFLContext
from ..common import check_type

default_cipher_list = "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:" \
                      "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-PSK-CHACHA20-POLY1305:" \
                      "ECDHE-ECDSA-AES128-CCM:ECDHE-ECDSA-AES256-CCM:ECDHE-ECDSA-CHACHA20-POLY1305"


class SSLConfig:
    """
    Define the ssl certificate config. If you want to start SSL, you need to configure the following parameters.
    The return value is used for the third input of `mindspore_federated.VerticalFederatedCommunicator` .

    Args:
        server_password (str): The password of the server certificate. Default: None.
        client_password (str): The password of the client certificate. Default: None.
        server_cert_path (str): The absolute path of the server certificate on the server. Default: None.
        client_cert_path (str): The absolute path of the client certificate on the server. Default: None.
        ca_cert_path (str): The absolute path of the root certificate on the server. Default: None.
        crl_path (str): The absolute path of the CRL certificate on the server. Default: None.
        cipher_list (str): The server supports the default encryption suite for ssl communication.
                           Default: default_cipher_list.
        cert_expire_warning_time_in_day (int): How many days before the certificate expires to start printing warning
                                               messages. Default: 90.
    """
    def __init__(self, server_password, client_password, server_cert_path=None, client_cert_path=None,
                 ca_cert_path=None, crl_path=None, cipher_list=default_cipher_list,
                 cert_expire_warning_time_in_day=90):
        """SSLConfig construct method"""
        check_type.check_str("server_password", server_password)
        check_type.check_str("client_password", client_password)
        self.server_password = server_password
        self.client_password = client_password

        if server_cert_path is not None:
            check_type.check_str("server_cert_path", server_cert_path)
            self.server_cert_path = server_cert_path
        if client_cert_path is not None:
            check_type.check_str("client_cert_path", client_cert_path)
            self.client_cert_path = client_cert_path
        if ca_cert_path is not None:
            check_type.check_str("ca_cert_path", ca_cert_path)
            self.ca_cert_path = ca_cert_path
        if crl_path is not None and crl_path != "":
            check_type.check_str("crl_path", crl_path)
            self.crl_path = crl_path
        else:
            self.crl_path = ""
        if cipher_list is not None:
            check_type.check_str("cipher_list", cipher_list)
            self.cipher_list = cipher_list
        if cert_expire_warning_time_in_day is not None:
            check_type.check_int("cert_expire_warning_time_in_day", cert_expire_warning_time_in_day)
            self.cert_expire_warning_time_in_day = cert_expire_warning_time_in_day


def init_ssl_config(ssl_config):
    """init ssl config method"""
    ctx = FLContext.get_instance()
    if ssl_config is not None:
        if not isinstance(ssl_config, SSLConfig):
            raise RuntimeError(f"Parameter 'ssl_config' should be instance of SSLConfig, but got {type(ssl_config)}")
        ctx.set_server_password(ssl_config.server_password)
        ctx.set_client_password(ssl_config.client_password)


def init_vertical_ssl_config(ssl_config):
    """init vertical federated ssl config method"""
    ctx = VFLContext.get_instance()
    if ssl_config is not None:
        if not isinstance(ssl_config, SSLConfig):
            raise RuntimeError(f"Parameter 'ssl_config' should be instance of SSLConfig, but got {type(ssl_config)}")
        ctx.set_server_password(ssl_config.server_password)
        ctx.set_client_password(ssl_config.client_password)
        ctx.set_server_cert_path(ssl_config.server_cert_path)
        ctx.set_client_cert_path(ssl_config.client_cert_path)
        ctx.set_ca_cert_path(ssl_config.ca_cert_path)
        ctx.set_crl_path(ssl_config.crl_path)
        ctx.set_cipher_list(ssl_config.cipher_list)
        ctx.set_cert_expire_warning_time_in_day(ssl_config.cert_expire_warning_time_in_day)


def init_vertical_enable_ssl(enable_ssl):
    """init vertical federated enable ssl"""
    ctx = VFLContext.get_instance()
    if not isinstance(enable_ssl, bool):
        raise RuntimeError(f"Parameter 'enable_ssl' should be instance of bool, but got {type(enable_ssl)}")
    ctx.set_enable_ssl(enable_ssl)
