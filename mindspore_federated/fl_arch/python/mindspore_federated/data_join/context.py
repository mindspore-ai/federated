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
        output_dir (str): The output directory.
        join_type (str): The data join type. Default: "psi".
        bucket_num (int): The number of buckets. Default: 1.
        store_type (str): The data store type. Default: "csv".
        primary_key (str): The primary key. Default: "oaid".
        thread_num (int): The thread number of psi. Default: 0.
        shard_num (int): The output number of each bucket when export. Default: 1.
    """

    def __init__(self, output_dir, join_type="psi", bucket_num=1, store_type="csv",
                 primary_key="oaid", thread_num=0, shard_num=1):
        self.output_dir = output_dir
        self.join_type = join_type
        self.bucket_num = bucket_num
        self.store_type = store_type
        self.primary_key = primary_key
        self.thread_num = thread_num
        self.shard_num = shard_num
