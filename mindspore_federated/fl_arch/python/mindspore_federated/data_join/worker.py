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
"""Worker in data join."""

import os
import logging
import yaml
import mmh3
from mindspore_federated.common import _checkparam as validator
from mindspore_federated.data_join.server import _DataJoinServer
from mindspore_federated.data_join.client import _DataJoinClient
from mindspore_federated.data_join.context import _WorkerRegister, _WorkerConfig
from mindspore_federated import VerticalFederatedCommunicator, ServerConfig
from mindspore_federated.data_join.store import DataSourceMgr
from mindspore_federated._mindspore_federated import VFLContext
from mindspore_federated.common import data_join_utils
from mindspore_federated.common.check_type import check_str
from mindspore_federated.startup.ssl_config import SSLConfig

from .io import export_mindrecord

SUPPORT_JOIN_TYPES = ("psi",)
SUPPORT_STORE_TYPES = ("csv",)


class _DivideKeyTobucket:
    """
    Divide key to bucket.

    Args:
        keys (list(str)): The keys need to be divided.
        bucket_num (int): The number of buckets.
    """

    def __init__(self, keys, bucket_num=64):
        self._bucket_num = bucket_num
        self._keys = keys

    def _get_bucket_id(self, key):
        return mmh3.hash(key) % self._bucket_num

    def get_buckets(self):
        """
        Returns:
            - buckets (list(str)): The list of ids in different buckets.
        """
        buckets = [list() for _ in range(self._bucket_num)]
        for key in self._keys:
            bucket_id = self._get_bucket_id(key)
            buckets[bucket_id].append(key)
        return buckets


class FLDataWorker:
    """
    Unlike horizontal federated learning, two participants (leader and follower) have the same sample space for training
    or inference in vertical federated learning. Therefore, the data intersection must be done collaboratively before
    both parties in vertical federated learning initiate training or inference. Both parties must read their respective
    original data and extract the ID (unique identifier of each data, and none of them is the same) corresponding to
    each data for intersection (i.e., finding the intersection). Then, both parties obtain features or tags from the
    original data based on the intersected IDs. Finally, each side exports the persistence file and reads the data in
    the reordering manner before subsequent training or inference. Data join worker is used to export data.

    Args:
        config (dict): the key/value of dict defined as below

            - role(str): Role of the worker, which must be set in both leader and follower. Supports [leader, follower].
              Default: "leader".
            - bucket_num(int): The number of buckets.
              If leader has set a valid value, the value set by follower will not be used. Default: 5.
            - store_type(str): The origin data store type. Now only support csv/mysql. Supports [csv, mysql].
              Default: "csv".
            - data_schema_path(str): Path of data schema file, which must be set in both leader and follower. User need
              to provide the column name and type of the data to be exported in the schema. The schema needs to be
              parsed as a two-level key-value dictionary. The key of the first-level dictionary is the column name, and
              the value is the second-level dictionary. The key of the second-level dictionary must be a string: type,
              and the value is the type of the exported data. Currently, the types support
              [int32, int64, float32, float64, string, bytes].  Default: "vfl/leader_schemal.yaml".
            - primary_key(str): The primary key.
              If leader has set a valid value, the value set by follower will not be used. Default: "oaid".
            - main_table_files(Union(list(str), str)): The raw data paths, which must be set in both leader and
              follower.  Default: "vfl/input/leader/".
            - mysql_host(str): Host where the database server is located. Default: "127.0.0.1".
            - mysql_port(int): MySQL port to use, usually use 3306. Default: 3306.
            - mysql_database(str): Database to use, None to not use a particular one. Default: "".
            - mysql_charset(str): Charset you want to use. Default: "".
            - mysql_user(str): Username to login mysql. Default: "test".
            - mysql_password(str): Password to login mysql. Default: "1234".
            - mysql_table_name(str): The table that contains origin data. Default: "test_data".
            - server_name(str): Local http server name, used for communication. Default: "leader_node".
            - http_server_address(str): Local IP and Port Address, which must be set in both leader and follower.
              Default: "127.0.0.1:1086".
            - remote_server_name(str): Remote http server name, used for communication. Default: "follower_node".
            - remote_server_address(str): Peer IP and Port Address, which must be set in both leader and follower.
              Default: "127.0.0.1:1087".
            - enable_ssl(bool): SSL mode enabled or disabled for communication. Supports [True, False]. Default: False.
            - server_password(str): The server password to decode the p12 file.
              For security please giving it in start command line. Default: "123456".
            - client_password(str): The client password to decode the p12 file.
              For security please giving it in start command line. Default: "123456".
            - server_cert_path(str): Certificate file path for server. Default: "server.p12".
            - client_cert_path(str): Certificate file path for client. Default: "client.p12".
            - ca_cert_path(str): CA server certificate file path. Default: "cacert.pem".
            - crl_path(str): CRL certificate file path. Default: "".
            - cipher_list(str): Encryption suite supported by ssl. Default: "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-
              AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305:
              ECDHE-PSK-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-CCM:ECDHE-ECDSA-AES256-CCM:ECDHE-ECDSA-CHACHA20-POLY1305".
            - cert_expire_warning_time_in_day(str): Warning time before the certificate expires. Default: 90.
            - join_type(str): The data join type.
              If leader has set a valid value, the value set by follower will not be used.
              Now only support "psi". Default: "psi".
            - thread_num(int): The thread number of psi. Default: 0.
            - output_dir(str): The output directory, which must be set in both leader and follower.
              Default: "vfl/output/leader/".
            - shard_num(int): The output number of each bucket when export.
              If leader has set a valid value, the value set by follower will not be used. Default: 1.

            More details refer to `vfl_data_join_config <https://e.gitee.com/mind_spore/repos/mindspore/federated/tree/master/tests/st/data_join/vfl/vfl_data_join_config.yaml>`_.


    Examples:
        >>> from mindspore_federated import FLDataWorker
        >>> from mindspore_federated.common.config import get_config
        >>>
        >>> current_dir = os.path.dirname(os.path.abspath(__file__))
        >>> args = get_config(os.path.join(current_dir, "vfl/vfl_data_join_config.yaml"))
        >>> dict_cfg = args.__dict__
        >>>
        >>> worker = FLDataWorker(config=dict_cfg)
        >>> worker.do_worker()
    """
    _communicator = None

    def __init__(self, config: dict):
        self._config = config

    def _join_func(self, input_vct, bucket_id):
        """
        Join function.

        Args:
            input_vct (list(str)): The keys need to be joined. The type of each key must be "str".
            bucket_id (int): The id of the bucket.

        Returns:
            - intersection_keys (list(str)): The intersection keys.
        """
        return self.data_join_obj.join_func(input_vct, bucket_id)

    def _export(self):
        """
        Export MindRecord by intersection keys.
        """
        keys = self._raw_data.keys()
        divide_key_to_bucket = _DivideKeyTobucket(bucket_num=self._worker_config.bucket_num, keys=keys)
        buckets = divide_key_to_bucket.get_buckets()
        shard_num = self._worker_config.shard_num
        export_count = 0
        for bucket_id, input_vct in enumerate(buckets):
            intersection_keys = self._join_func(input_vct, bucket_id + 1)
            if not intersection_keys:
                continue
            file_name = "mindrecord_{}_".format(bucket_id) if shard_num > 1 else "mindrecord_{}".format(bucket_id)
            output_file_name = os.path.join(self._worker_config.output_dir, file_name)
            export_mindrecord(output_file_name, self._raw_data, intersection_keys, shard_num=shard_num)
            export_count += 1
        if export_count == 0:
            raise ValueError("The intersection_keys of all buckets is empty")

    def _get_data_source(self):
        """
        create data source by config
        """
        with open(self._config['data_schema_path'], "r") as f:
            self._schema = yaml.load(f, yaml.Loader)

        cls_data_source = DataSourceMgr.get_data_source_cls(self._worker_config.store_type)
        if cls_data_source is not None:
            self._raw_data = cls_data_source(store_type=self._worker_config.store_type,
                                             primary_key=self._worker_config.primary_key,
                                             schema=self._schema,
                                             config=self._config)
        else:
            raise ValueError("Unsupported Data Source type " + self._worker_config.store_type)

        self._raw_data.verify()
        self._raw_data.load_raw_data()

    def _get_data_joiner(self):
        """
        create data joiner by config
        """
        role = self._config['role']
        if role == "leader":
            self.data_join_obj = _DataJoinServer(self._worker_config, FLDataWorker._communicator)
        elif role == "follower":
            self.data_join_obj = _DataJoinClient(self._worker_config, FLDataWorker._communicator)
        else:
            raise ValueError("role must be \"leader\" or \"follower\"")

    def _verify_worker_config(self):
        """
        verify worker config
        """
        validator.check_string(self._worker_config.join_type, SUPPORT_JOIN_TYPES, arg_name="join_type")
        validator.check_int_range(self._worker_config.bucket_num, 1, 1000000, validator.INC_BOTH, arg_name="bucket_num")
        validator.check_string(self._worker_config.store_type, SUPPORT_STORE_TYPES, arg_name="store_type")
        check_str(arg_name="primary_key", str_val=self._worker_config.primary_key)
        validator.check_non_negative_int(self._worker_config.thread_num, arg_name="thread_num")
        validator.check_int_range(self._worker_config.shard_num, 1, 1000, validator.INC_BOTH, arg_name="shard_num")
        if self._worker_config.shard_num * self._worker_config.bucket_num > 4096:
            logging.warning('The maximum number of files read by MindData is 4096. It is recommended that the value of '
                            'shard_num * bucket_num be smaller than 4096. Actually, the value is: %d',
                            self._worker_config.shard_num * self._worker_config.bucket_num)
        check_str(arg_name="output_dir", str_val=self._worker_config.output_dir)
        if not os.path.isdir(self._worker_config.output_dir):
            raise ValueError("output_dir: {} is not a directory.".format(self._worker_config.output_dir))

    def _negotiate_hyper_params(self):
        """
        negotiate hyperparameters
        The hyperparameters include:

        - primary_key (str)
        - bucket_num (int)
        - shard_num (int)
        - join_type (str)
        """
        self._worker_config = _WorkerConfig(
            output_dir=self._config['output_dir'],
            primary_key=self._config['primary_key'],
            bucket_num=self._config['bucket_num'],
            store_type=self._config['store_type'],
            shard_num=self._config['shard_num'],
            join_type=self._config['join_type'],
            thread_num=self._config['thread_num'],
        )
        role = self._config['role']
        if role == "leader":
            self._verify_worker_config()
            ctx = VFLContext.get_instance()
            worker_config_item_py = data_join_utils.worker_config_to_pybind_obj(self._worker_config)
            ctx.set_worker_config(worker_config_item_py)
            FLDataWorker._communicator.data_join_wait_for_start()
        elif role == "follower":
            worker_register = _WorkerRegister(role)
            primary_key, bucket_num, shard_num, join_type = FLDataWorker._communicator.send_register(
                self._config['remote_server_name'],
                worker_register=worker_register)
            self._worker_config.primary_key = primary_key
            self._worker_config.bucket_num = bucket_num
            self._worker_config.shard_num = shard_num
            self._worker_config.join_type = join_type
            self._verify_worker_config()
        else:
            raise ValueError("role must be \"leader\" or \"follower\"")

    def _create_communicator(self):
        """
        create communicator for data join
        communicator will be used both at data join && model train
        """
        if FLDataWorker._communicator is not None:
            return

        # create communicator for data join
        http_server_config = ServerConfig(server_name=self._config['server_name'],
                                          server_address=self._config['http_server_address'])
        remote_server_config = ServerConfig(server_name=self._config['remote_server_name'],
                                            server_address=self._config['remote_server_address'])

        enable_ssl = self._config['enable_ssl']
        ssl_config = None
        if isinstance(enable_ssl, bool) and enable_ssl:
            ssl_config = SSLConfig(
                self._config['server_password'],
                self._config['client_password'],
                self._config['server_cert_path'],
                self._config['client_cert_path'],
                self._config['ca_cert_path'],
                self._config['crl_path'],
                self._config['cipher_list'],
                self._config['cert_expire_warning_time_in_day']
            )
        FLDataWorker._communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
                                                                   remote_server_config=remote_server_config,
                                                                   enable_ssl=enable_ssl,
                                                                   ssl_config=ssl_config)
        FLDataWorker._communicator.launch()

    def communicator(self):
        """
        If you want to use same communicator in both data join and vfl model training, you can call this function to
        obtain the communicator instance.
        """
        return FLDataWorker._communicator

    def do_worker(self):
        """
        Execute data join worker according to the config.

        Examples:
            >>> cfg_dict = dict()
            >>> cfg_dict['server_name'] = "james"
            >>> cfg_dict['http_server_address'] = "127.0.0.1:1984"
            >>> cfg_dict['remote_server_name'] = "kobe"
            >>> cfg_dict['remote_server_address'] = "127.0.0.1:1230"
            >>> cfg_dict['enable_ssl'] = True
            >>> cfg_dict['main_table_files'] = "leader_data.csv"
            >>> cfg_dict['role'] = "leader"
            >>> cfg_dict['output_dir'] = "leader/"
            >>> cfg_dict['data_schema_path'] = "leader_schema.yaml"
            >>> cfg_dict['shard_num'] = 2
            >>> work = FLDataWorker(cfg_dict)
            >>> work.do_worker()
        """
        self._create_communicator()

        # negotiate params
        self._negotiate_hyper_params()

        # get data source
        self._get_data_source()

        # get data joiner
        self._get_data_joiner()

        #
        self._export()
