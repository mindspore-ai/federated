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
"""Worker in data join."""

import os
import yaml
import mmh3
from mindspore._checkparam import Validator, Rel
from mindspore_federated.data_join.server import _DataJoinServer
from mindspore_federated.data_join.client import _DataJoinClient
from mindspore_federated.data_join.context import _WorkerRegister, _WorkerConfig
from mindspore_federated import VerticalFederatedCommunicator, ServerConfig
from .io import export_mindrecord

SUPPORT_JOIN_TYPES = ("psi",)
SUPPORT_STORE_TYPES = ("csv",)
SUPPORT_TYPES = ("int32", "int64", "float32", "float64", "string", "bytes")
SUPPORT_ARRAY_TYPES = ("int32", "int64", "float32", "float64")


def _check_str(arg_value, arg_name=None, prim_name=None):
    if not isinstance(arg_value, str):
        prim_name = f"For '{prim_name}', the" if prim_name else 'The'
        arg_name = f"'{arg_name}'" if arg_name else 'input value'
        raise TypeError(f"{prim_name} {arg_name} must be a str, but got {type(arg_value).__name__}.")
    return arg_value


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
    Data join worker.

    Args:
        role (str): Role of the worker, which must be set in both leader and follower. Supports ["leader", "follower"].
        main_table_files (Union(list(str), str): The raw data paths, which must be set in both leader and follower.
        output_dir (str): The output directory, which must be set in both leader and follower.
        data_schema_path (str): Path of data schema file, which must be set in both leader and follower. User need to
            provide the column name and type of the data to be exported in the schema.
            The schema needs to be parsed as a two-level key-value dictionary.
            The key of the first-level dictionary is the column name, and the value is the second-level dictionary.
            The key of the second-level dictionary must be a string: "type", and the value is the type of the
            exported data.
            Currently, the types support ["int32", "int64", "float32", "float64", "string", "bytes"].
        http_server_address (str): Local IP and Port Address, which must be set in both leader and follower.
        remote_server_address (str): Peer IP and Port Address, which must be set in both leader and follower.
        primary_key (str): The primary key. The value set by leader is used, and the value set by follower is invalid.
            Default: "oaid".
        bucket_num (int): The number of buckets. The value set by leader is used, and the value set by follower is
            invalid. Default: 1.
        store_type (str): The data store type. Default: "csv".
        shard_num (int): The output number of each bucket when export. The value set by leader is used, and the value
            set by follower is invalid. Default: 1.
        join_type (str): The data join type. The value set by leader is used, and the value set by follower is
            invalid. Default: "psi".
        thread_num (int): The thread number of psi. Default: 0.

    Examples:
        >>> worker = FLDataWorker(role="leader",
        ...                       main_table_files=["input/leader_data_0.csv", "input/leader_data_1.csv"],
        ...                       output_dir="output/leader/",
        ...                       data_schema_path="leader_schema.yaml",
        ...                       http_server_address="127.0.0.1:1086",
        ...                       remote_server_address="127.0.0.1:1087",
        ...                       primary_key="oaid",
        ...                       bucket_num=5,
        ...                       store_type="csv",
        ...                       shard_num=1,
        ...                       join_type="psi",
        ...                       thread_num=0,
        ...                       )
        >>> worker.export()
    """

    def __init__(self,
                 role,
                 main_table_files,
                 output_dir,
                 data_schema_path,
                 http_server_address,
                 remote_server_address,
                 primary_key="oaid",
                 bucket_num=5,
                 store_type="csv",
                 shard_num=1,
                 join_type="psi",
                 thread_num=0,
                 ):
        self._role = role
        self._worker_config = _WorkerConfig(
            main_table_files=main_table_files,
            output_dir=output_dir,
            primary_key=primary_key,
            bucket_num=bucket_num,
            store_type=store_type,
            shard_num=shard_num,
            join_type=join_type,
            thread_num=thread_num,
        )
        self._data_schema_path = data_schema_path
        if self._role == "leader":
            server_name = "server"
            peer_server_name = "client"
        elif self._role == "follower":
            server_name = "client"
            peer_server_name = "server"
        else:
            raise ValueError("role must be \"leader\" or \"follower\"")
        with open(self._data_schema_path, "r") as f:
            self._schema = yaml.load(f, yaml.Loader)
        self._verify()

        http_server_config = ServerConfig(server_name=server_name, server_address=http_server_address)
        remote_server_config = ServerConfig(server_name=peer_server_name, server_address=remote_server_address)
        vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
                                                              remote_server_config=remote_server_config)
        self._vertical_communicator = vertical_communicator
        vertical_communicator.launch()
        if role == "leader":
            self.data_join_obj = _DataJoinServer(self._worker_config, self._vertical_communicator)
            self.data_join_obj.launch()
        elif role == "follower":
            worker_register = _WorkerRegister(self._role)
            self.data_join_obj = _DataJoinClient(self._worker_config, self._vertical_communicator, worker_register)
            self._worker_config = self.data_join_obj.launch()
        else:
            raise ValueError("role must be \"leader\" or \"follower\"")
        self._verify()

    def _verify(self):
        """
        Verify hyper parameters and schema.
        """
        self._verify_main_table_files()
        _check_str(self._worker_config.output_dir, arg_name="output_dir")
        if not os.path.isdir(self._worker_config.output_dir):
            raise ValueError("output_dir: {} is not a directory.".format(self._worker_config.output_dir))
        Validator.check_string(self._worker_config.join_type, SUPPORT_JOIN_TYPES, arg_name="join_type")
        Validator.check_int_range(self._worker_config.bucket_num, 1, 1000000, Rel.INC_BOTH, arg_name="bucket_num")
        Validator.check_string(self._worker_config.store_type, SUPPORT_STORE_TYPES, arg_name="store_type")

        _check_str(self._worker_config.primary_key, arg_name="primary_key")
        Validator.check_non_negative_int(self._worker_config.thread_num, arg_name="thread_num")
        Validator.check_int_range(self._worker_config.shard_num, 1, 1000, Rel.INC_BOTH, arg_name="shard_num")
        self._verify_schema()

    def _verify_main_table_files(self):
        """
        Verify main_table_files.
        """
        main_table_files = self._worker_config.main_table_files
        if isinstance(main_table_files, list):
            for main_table_file in main_table_files:
                if not os.path.isfile(main_table_file):
                    raise ValueError("{} in main_table_files is not a file.".format(main_table_file))
        elif isinstance(main_table_files, str):
            if not os.path.exists(main_table_files):
                raise ValueError("main_table_files: {} is not exist.".format(main_table_files))
            if os.path.isdir(main_table_files):
                for main_table_file in os.listdir(main_table_files):
                    main_table_file = os.path.join(main_table_files, main_table_file)
                    if not os.path.isfile(main_table_file):
                        raise ValueError("{} in main_table_files is not a file.".format(main_table_file))
        else:
            raise TypeError("main_table_files must be list or str, but get {}".format(type(main_table_files)))

    def _verify_schema(self):
        """
        Verify schema.
        """
        if isinstance(self._schema, dict):
            for key in self._schema:
                _check_str(key, arg_name="column name")

                shape = self._schema[key].get("shape")
                data_type = self._schema[key].get("type")

                if shape is not None:
                    if isinstance(shape, list):
                        raise TypeError("shape must be list, but get {}".format(type(shape)))
                else:
                    shape = (1,)

                if data_type is not None:
                    if len(shape) == 1:
                        Validator.check_string(data_type, SUPPORT_TYPES, arg_name="data type")
                    else:
                        Validator.check_string(data_type, SUPPORT_ARRAY_TYPES, arg_name="array data type")
        else:
            raise TypeError("schema must be dict, but get {}".format(type(self._schema)))

    def _load_raw_data(self):
        """
        Load data from the file system. Only support "csv" currently.

        Returns:
            - raw_data (BaseData): The raw data.
        """
        if self._worker_config.store_type == "csv":
            from .store import PandasData
            raw_data = PandasData(None, primary_key=self._worker_config.primary_key, schema=self._schema)
            main_table_files = self._worker_config.main_table_files
            if isinstance(main_table_files, list):
                for main_table_file in main_table_files:
                    raw_data.load_raw_data(main_table_file)
            elif isinstance(main_table_files, str):
                if os.path.isdir(main_table_files):
                    index = 0
                    for main_table_file in os.listdir(main_table_files):
                        index += 1
                        main_table_file = os.path.join(main_table_files, main_table_file)
                        raw_data.load_raw_data(main_table_file)
                else:
                    raw_data.load_raw_data(main_table_files)
            else:
                raise TypeError("main_table_files must be list or str, but get {}".format(
                    type(main_table_files)))
        else:
            raise ValueError("store type: {} is not support currently".format(self._worker_config.store_type))
        return raw_data

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

    def export(self):
        """
        Export MindRecord by intersection keys.
        """
        raw_data = self._load_raw_data()
        keys = raw_data.keys()
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
            export_mindrecord(output_file_name, raw_data, intersection_keys, shard_num=shard_num)
            export_count += 1
        if export_count == 0:
            raise ValueError("The intersection_keys of all buckets is empty")
