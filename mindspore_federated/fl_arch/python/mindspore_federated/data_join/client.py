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


class _DataJoinClient:
    """
    Data join client.
    """

    def __init__(self, worker_config, vertical_communicator):
        """
        Args:
            worker_config (_WorkerConfig): The config of worker.
        """
        self._worker_config = worker_config
        self._vertical_communicator = vertical_communicator
        self._target_server_name = vertical_communicator.remote_server_config().server_name

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
        if self._worker_config.join_type == "psi":
            thread_num = self._worker_config.thread_num
            intersection_keys = RunPSI(input_vct, "client", self._target_server_name, bucket_id, thread_num)
            return intersection_keys
        raise ValueError("join type: {} is not support currently".format(self._worker_config.join_type))
