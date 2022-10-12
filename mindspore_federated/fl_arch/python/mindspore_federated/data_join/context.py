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
"""Worker Context."""

import yaml


class _WorkerRegister:
    """
    Worker register msg.

    Args:
        worker_name (str): The register worker name.
    """

    def __init__(self, worker_name):
        self.worker_name = worker_name


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
        self.thread_num = worker_config_dict.get("thread_num", 0)
        self.shard_num = worker_config_dict.get("shard_num", 1)
