import mmh3
import yaml
import logging
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
        file_name (str): The prefix of output file name.
        join_type (str): The data join type. Default: "psi".
        bin_num (int): The number of bins. Default: 1.
        store_type (str): The data store type. Default: "csv".
        primary_key (str): The primary key. Default: "oaid".
        http_server_address (str): The address of local address. Default: "127.0.0.1:8080".
        remote_server_address (str): The address of remote address. Default: "127.0.0.1:18080".
        thread_num (int): The thread number of psi. The prefix of output file name. Default: 0.
        shard_num (int): The output number of each bin when export. Default: 1.
        overwrite (bool): If overwrite files when export. Default: False.
    """

    def __init__(self, worker_config_path):
        with open(worker_config_path, "r") as f:
            worker_config_dict = yaml.load(f, yaml.Loader)
        if "main_table_files" in worker_config_dict:
            self.main_table_files = worker_config_dict.get("main_table_files")
        else:
            raise ValueError("main_table_files must be in worker_config")
        if "output_file_name" in worker_config_dict:
            self.file_name = worker_config_dict.get("output_file_name")
        else:
            raise ValueError("output_file_name must be in worker_config")
        self.join_type = worker_config_dict.get("join_type", "psi")
        self.bin_num = worker_config_dict.get("bin_num", 1)
        self.store_type = worker_config_dict.get("store_type", "csv")
        self.primary_key = worker_config_dict.get("primary_key", "oaid")
        self.http_server_address = worker_config_dict.get("http_server_address", "127.0.0.1:8080")
        self.remote_server_address = worker_config_dict.get("remote_server_address", "127.0.0.1:18080")
        self.thread_num = worker_config_dict.get("thread_num", 0)
        self.shard_num = worker_config_dict.get("shard_num", 1)
        self.overwrite = worker_config_dict.get("overwrite", False)


class _DivideKeyTobin:
    """
    Divide key to bin.

    Args:
        keys (list(str)): The keys need to be divided.
        bin_num (int): The number of bins.
    """

    def __init__(self, keys, bin_num=64):
        self._bin_num = bin_num
        self._keys = keys

    def _get_bin_id(self, key):
        return mmh3.hash(key) % self._bin_num

    def get_bins(self):
        """
        Returns:
            - bins (list(str)): The list of ids in different bins.
        """
        bins = [list() for _ in range(self._bin_num)]
        for key in self._keys:
            bin_id = self._get_bin_id(key)
            bins[bin_id].append(key)
        return bins


class FLDataWorker:
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
        if not isinstance(self._worker_config.file_name, str):
            raise TypeError("file_name must be str, but get {}".format(type(self._worker_config.file_name)))
        if isinstance(self._worker_config.join_type, str):
            if self._worker_config.join_type not in SUPPORT_JOIN_TYPES:
                raise ValueError("join_type must be in {}".format(str(SUPPORT_JOIN_TYPES)))
        else:
            raise TypeError("join_type must be str, but get {}".format(type(self._worker_config.join_type)))
        if isinstance(self._worker_config.bin_num, int):
            if self._worker_config.bin_num < 0:
                raise ValueError("bin_num must be bigger than 0, but get {}".format(self._worker_config.bin_num))
        else:
            raise TypeError("bin_num must be int, but get {}".format(type(self._worker_config.bin_num)))
        if isinstance(self._worker_config.store_type, str):
            if self._worker_config.store_type not in SUPPORT_STORE_TYPES:
                raise ValueError("store_type must be in {}".format(str(SUPPORT_STORE_TYPES)))
        else:
            raise TypeError("store_type must be str, but get {}".format(type(self._worker_config.store_type)))
        if not isinstance(self._worker_config.primary_key, str):
            raise TypeError("primary_key must be str, but get {}".format(type(self._worker_config.primary_key)))
        if not isinstance(self._worker_config.http_server_address, str):
            raise TypeError(
                "http_server_address must be str, but get {}".format(type(self._worker_config.http_server_address)))
        if not isinstance(self._worker_config.remote_server_address, str):
            raise TypeError(
                "remote_server_address must be str, but get {}".format(type(self._worker_config.remote_server_address)))
        if not isinstance(self._worker_config.thread_num, int):
            raise TypeError("thread_num must be int, but get {}".format(type(self._worker_config.thread_num)))
        elif self._worker_config.thread_num < 0:
            raise ValueError(
                "thread_num must be bigger than 0, but get {}".format(self._worker_config.thread_num))
        if not isinstance(self._worker_config.shard_num, int):
            raise TypeError("shard_num must be int, but get {}".format(type(self._worker_config.shard_num)))
        elif self._worker_config.shard_num < 1 or self._worker_config.shard_num > 1000:
            raise ValueError("shard_num should be between [1, 1000], but get {}".format(self._worker_config.shard_num))
        if not isinstance(self._worker_config.overwrite, bool):
            raise TypeError("overwrite must be bool, but get {}".format(type(self._worker_config.overwrite)))
        if isinstance(self._schema, dict):
            for key in self._schema:
                if not isinstance(key, str):
                    raise TypeError("field name: {} must be str, but get {}".format(key, type(key)))

                shape = self._schema[key].get("shape")
                data_type = self._schema[key].get("type")

                if shape is not None:
                    if isinstance(shape, list):
                        raise TypeError("shape: must be list, but get {}".format(shape, type(shape)))
                else:
                    shape = (1,)

                if len(shape) == 1:
                    if data_type is not None and data_type not in SUPPORT_TYPES:
                        raise ValueError("type must be in {}, but get {}".format(str(SUPPORT_TYPES), data_type))
                else:
                    if data_type is not None and data_type not in SUPPORT_ARRAY_TYPES:
                        raise ValueError("type must be in {}, but get {}".format(str(SUPPORT_ARRAY_TYPES), data_type))
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
                    raw_data.merge(df)
            elif isinstance(self._worker_config.main_table_files, str):
                df = pd.read_csv(self._worker_config.main_table_files)
                raw_data = PandasData(df, primary_key=self._worker_config.primary_key, schema=self._schema)
            else:
                raise TypeError("main_table_files must be list or str, but get {}".format(
                    type(self._worker_config.main_table_files)))
        else:
            raise ValueError("store type: {} is not support currently".format(self._worker_config.store_type))
        return raw_data

    def _join_func(self, input_vct, bin_id):
        """
        Join function.

        Args:
            input_vct (list(str)): The keys need to be joined. The type of each key must be "str".
            bin_id (int): The id of the bin.

        Returns:
            - intersection_keys (list(str)): The intersection keys.
        """
        return self.communicator.join_func(input_vct, bin_id)

    def export(self):
        """
        Export MindRecord by intersection keys.
        """
        raw_data = self._load_raw_data()
        keys = raw_data.keys()
        divide_key_to_bin = _DivideKeyTobin(bin_num=self._worker_config.bin_num, keys=keys)
        bins = divide_key_to_bin.get_bins()
        shard_num = self._worker_config.shard_num
        overwrite = self._worker_config.overwrite
        export_count = 0
        for bin_id, input_vct in enumerate(bins):
            intersection_keys = self._join_func(input_vct, bin_id)
            if len(intersection_keys) == 0:
                logging.debug("The intersection_keys of bin {} is empty".format(bin_id))
                continue
            output_file_name = "{}_{}_".format(self._worker_config.file_name, bin_id) if shard_num > 1 else \
                "{}_{}".format(self._worker_config.file_name, bin_id)
            export_mindrecord(output_file_name, raw_data, intersection_keys, shard_num=shard_num, overwrite=overwrite)
            export_count += 1
        if export_count == 0:
            raise ValueError("The intersection_keys of all bins is empty")
