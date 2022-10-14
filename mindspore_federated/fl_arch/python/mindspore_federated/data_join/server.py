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
"""Communicator server in data join."""
from mindspore_federated._mindspore_federated import RunPSI
from mindspore_federated._mindspore_federated import VFLContext
from mindspore_federated.common import data_join_utils


def server_psi_is_ready(bucket_id):
    with open("server_psi_{}.txt".format(bucket_id), "w") as f:
        f.write("")


class _DataJoinServer:
    """
    Data join server.
    """

    def __init__(self, worker_config, vertical_communicator):
        """
        Args:
            worker_config (_WorkerConfig): The config of worker.
            target_server_name (str): The target communicator server name.
        """
        self._worker_config = worker_config
        self._vertical_communicator = vertical_communicator
        self._target_server_name = vertical_communicator.remote_server_config().server_name

        ctx = VFLContext.get_instance()
        worker_config_item_py = data_join_utils.worker_config_to_pybind_obj(self._worker_config)
        ctx.set_worker_config(worker_config_item_py)

    def launch(self):
        """
        Negotiate hyper parameters with client.
        """
        return self._wait_for_negotiated()

    def _wait_for_negotiated(self):
        """
        Wait for hyper parameters request from client.

        The hyper parameters include:
            primary_key (str)
            bucket_num (int)
            shard_num (int)
            join_type (str)

        Returns:
            - worker_config (_WorkerConfig): The config of worker.
        """
        return self._vertical_communicator.data_join_wait_for_start()

    def join_func(self, input_vct, bucket_id):
        """
        Join function.

        Args:
            input_vct (list(str)): The keys need to be joined. The type of each key must be "str".
            bucket_id (int): The id of the bucket.
            target_server_name (str): The target communicator server name.

        Returns:
            - intersection_keys (list(str)): The intersection keys.

        Raises:
            ValueError: If the join type is not supported.
        """
        server_psi_is_ready(bucket_id)
        if self._worker_config.join_type == "psi":
            thread_num = self._worker_config.thread_num
            intersection_keys = RunPSI(input_vct, "server", self._target_server_name, bucket_id, thread_num)
            return intersection_keys
        raise ValueError("join type: {} is not support currently".format(self._worker_config.join_type))
