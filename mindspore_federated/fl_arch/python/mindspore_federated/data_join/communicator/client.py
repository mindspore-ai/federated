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
"""Communicator client in data join."""
from mindspore_federated._mindspore_federated import RunPSI


class _SimplifiedWorkerConfig:
    def __init__(self, worker_config_dict):
        self.join_type = worker_config_dict["join_type"]
        self.bucket_num = worker_config_dict["bucket_num"]
        self.primary_key = worker_config_dict["primary_key"]
        self.shard_num = worker_config_dict["shard_num"]


def request_params(http_server_address, remote_server_address):
    """
    fake communication
    """
    print("http_server_address:", http_server_address)
    print("remote_server_address:", remote_server_address)
    import yaml
    import os
    while True:
        if os.path.exists("server_psi_yaml.yaml"):
            break
    with open("server_psi_yaml.yaml", "r") as f:
        worker_config_dict = yaml.safe_load(stream=f)
    worker_config = _SimplifiedWorkerConfig(worker_config_dict)
    os.remove("server_psi_yaml.yaml")
    return worker_config


def wait_util_server_psi_is_ready(bucket_id):
    import os
    while True:
        if os.path.exists("server_psi_{}.txt".format(bucket_id)):
            os.remove("server_psi_{}.txt".format(bucket_id))
            break


class _DataJoinClient:
    """
    Data join client.
    """

    def __init__(self, worker_config):
        """
        Args:
            worker_config (_WorkerConfig): The config of worker.
        """
        self._worker_config = worker_config

    def wait_for_negotiated(self):
        """
        Negotiate hyper parameters with server.
        """
        return self._request_hyper_params()

    def _request_hyper_params(self):
        """
        Request and verify hyper parameters from server. Overwrite hyper parameters in local worker config.

        The hyper parameters include:
            primary_key (str)
            bucket_num (int)
            shard_num (int)
            join_type (str)

        Returns:
            - worker_config (_WorkerConfig): The config of worker.
        """
        # TODO: send the above hyper parameters to client
        http_server_address = self._worker_config.http_server_address
        remote_server_address = self._worker_config.remote_server_address
        worker_config = request_params(http_server_address, remote_server_address)
        self._worker_config.primary_key = worker_config.primary_key
        self._worker_config.bucket_num = worker_config.bucket_num
        self._worker_config.shard_num = worker_config.shard_num
        self._worker_config.join_type = worker_config.join_type
        return self._worker_config

    def join_func(self, input_vct, bucket_id):
        """
        Join function.

        Args:
            input_vct (list(str)): The keys need to be joined. The type of each key must be "str".
            bucket_id (int): The id of the bucket.

        Returns:
            - intersection_keys (list(str)): The intersection keys.

        Raises:
            ValueError: If the join type is not supported.
        """
        wait_util_server_psi_is_ready(bucket_id)
        if self._worker_config.join_type == "psi":
            thread_num = self._worker_config.thread_num
            http_server_address = self._worker_config.http_server_address
            remote_server_address = self._worker_config.remote_server_address
            intersection_keys = RunPSI(input_vct, "client", http_server_address, remote_server_address,
                                       thread_num, bucket_id)
            return intersection_keys
        raise ValueError("join type: {} is not support currently".format(self._worker_config.join_type))
