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
from .io import export_mindrecord
from .communicator import _DataJoinServer, _DataJoinClient

SUPPORT_JOIN_TYPES = ("psi",)
SUPPORT_STORE_TYPES = ("csv",)
SUPPORT_TYPES = ("int32", "int64", "float32", "float64", "string", "bytes")
SUPPORT_ARRAY_TYPES = ("int32", "int64", "float32", "float64")


class _WorkerConfig:
    """
    Config of worker.

    Args:
        worker_config_path (str): The config path. The content of the file corresponding to the path must be in the
            YAML format.

    Attribute:
        main_table_files (Union(list(str), str): The raw data paths.
        output_dir (str): The output directory.
        join_type (str): The data join type. Default: "psi".
        bucket_num (int): The number of buckets. Default: 1.
        store_type (str): The data store type. Default: "csv".
        primary_key (str): The primary key. Default: "oaid".
        http_server_address (str): The address of local address. Default: "127.0.0.1:8080".
        remote_server_address (str): The address of remote address. Default: "127.0.0.1:18080".
        thread_num (int): The thread number of psi. The prefix of output file name. Default: 0.
        shard_num (int): The output number of each bucket when export. Default: 1.
    """

    def __init__(self, worker_config_path):
        with open(worker_config_path, "r") as f:
            worker_config_dict = yaml.load(f, yaml.Loader)
        if "main_table_files" in worker_config_dict:
            self.main_table_files = worker_config_dict.get("main_table_files")
        else:
            raise ValueError("main_table_files must be in worker_config")
        if "output_dir" in worker_config_dict:
            self.output_dir = worker_config_dict.get("output_dir")
        else:
            raise ValueError("output_dir must be in worker_config")
        self.join_type = worker_config_dict.get("join_type", "psi")
        self.bucket_num = worker_config_dict.get("bucket_num", 1)
        self.store_type = worker_config_dict.get("store_type", "csv")
        self.primary_key = worker_config_dict.get("primary_key", "oaid")
        self.http_server_address = worker_config_dict.get("http_server_address", "127.0.0.1:8080")
        self.remote_server_address = worker_config_dict.get("remote_server_address", "127.0.0.1:18080")
        self.thread_num = worker_config_dict.get("thread_num", 0)
        self.shard_num = worker_config_dict.get("shard_num", 1)


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
    """
    def __init__(self,
                 role,
                 worker_config_path,
                 data_schema_path,
                 ):
        """
        Data join worker.

        Args:
            role (str): mark "leader" of "follower" role of the worker
            worker_config_path (str):
            data_schema_path (str):
        """
        self._role = role
        self._worker_config = _WorkerConfig(worker_config_path)
        self._data_schema_path = data_schema_path
        with open(self._data_schema_path, "r") as f:
            self._schema = yaml.load(f, yaml.Loader)
        self._verify()
        if role == "leader":
            self.communicator = _DataJoinServer(self._worker_config)
        elif role == "follower":
            self.communicator = _DataJoinClient(self._worker_config)
        else:
            raise ValueError("role must be \"leader\" or \"follower\"")
        self._worker_config = self.communicator.wait_for_negotiated()
        self._verify()

    def _verify(self):
        """
        Verify hyper parameters and schema.
        """
        main_table_files = self._worker_config.main_table_files
        if not isinstance(main_table_files, list) and not isinstance(main_table_files, str):
            raise TypeError("main_table_files must be list or str, but get {}".format(type(main_table_files)))
        _check_str(self._worker_config.output_dir, arg_name="output_dir")
        Validator.check_string(self._worker_config.join_type, SUPPORT_JOIN_TYPES, arg_name="join_type")
        Validator.check_int_range(self._worker_config.bucket_num, 1, 1000000, Rel.INC_BOTH, arg_name="bucket_num")
        Validator.check_string(self._worker_config.store_type, SUPPORT_STORE_TYPES, arg_name="store_type")

        _check_str(self._worker_config.primary_key, arg_name="primary_key")
        _check_str(self._worker_config.http_server_address, arg_name="http_server_address")
        _check_str(self._worker_config.remote_server_address, arg_name="remote_server_address")
        Validator.check_non_negative_int(self._worker_config.thread_num, arg_name="thread_num")
        Validator.check_int_range(self._worker_config.shard_num, 1, 1000, Rel.INC_BOTH, arg_name="shard_num")
        self._verify_schema()

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
            import pandas as pd
            from .store import PandasData
            if isinstance(self._worker_config.main_table_files, list):
                raw_data = PandasData(None, primary_key=self._worker_config.primary_key, schema=self._schema)
                for main_table_file in self._worker_config.main_table_files:
                    df = pd.read_csv(main_table_file)
                    df[self._worker_config.primary_key] = df[self._worker_config.primary_key].astype("str")
                    raw_data.merge(df)
            elif isinstance(self._worker_config.main_table_files, str):
                df = pd.read_csv(self._worker_config.main_table_files)
                df[self._worker_config.primary_key] = df[self._worker_config.primary_key].astype("str")
                raw_data = PandasData(df, primary_key=self._worker_config.primary_key, schema=self._schema)
            else:
                raise TypeError("main_table_files must be list or str, but get {}".format(
                    type(self._worker_config.main_table_files)))
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
        return self.communicator.join_func(input_vct, bucket_id)

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
