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
from mindspore_federated.common import data_join_utils


class _SimplifiedWorkerConfig:
    def __init__(self, worker_config_dict):
        self.join_type = worker_config_dict["join_type"]
        self.bucket_num = worker_config_dict["bucket_num"]
        self.primary_key = worker_config_dict["primary_key"]
        self.shard_num = worker_config_dict["shard_num"]


def request_params():
    """
    fake communication
    """
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

    def __init__(self, worker_config, vertical_communicator, worker_register):
        """
        Args:
            worker_config (_WorkerConfig): The config of worker.
        """
        self._worker_config = worker_config
        self._worker_register = worker_register
        self._vertical_communicator = vertical_communicator
        self._target_server_name = vertical_communicator.remote_server_config().server_name

    def launch(self):
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
        return self._register()

    def _register(self):
        """
        Register to server worker.
        """
        worker_register_item_py = data_join_utils.worker_register_to_pybind_obj(self._worker_register)
        worker_config_item_py = self._vertical_communicator.send_register(
            self._target_server_name,
            worker_register_item_py=worker_register_item_py)
        primary_key, bucket_num, shard_num, join_type = \
            data_join_utils.pybind_obj_to_worker_config(worker_config_item_py)
        self._worker_config.primary_key = primary_key
        self._worker_config.bucket_num = bucket_num
        self._worker_config.shard_num = shard_num
        self._worker_config.join_type = join_type
        return self._worker_config

    def join_func(self, input_vct, bucket_id):
        """
        Join function.

        Args:
            input_vct (list(str)): The keys need to be joined. The type of each key must be "str".
            bucket_id (int): The id of the bucket.
            target_server_name (str): The target communicator server name:

        Returns:
            - intersection_keys (list(str)): The intersection keys.

        Raises:
            ValueError: If the join type is not supported.
            :param target_server_name:
        """
        wait_util_server_psi_is_ready(bucket_id)
        if self._worker_config.join_type == "psi":
            thread_num = self._worker_config.thread_num
            intersection_keys = RunPSI(input_vct, "client", self._target_server_name, bucket_id, thread_num)
            return intersection_keys
        raise ValueError("join type: {} is not support currently".format(self._worker_config.join_type))