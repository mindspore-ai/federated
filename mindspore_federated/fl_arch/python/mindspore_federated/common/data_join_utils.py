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
"""Essential tools to modeling the data join process."""
from mindspore_federated import log as logger
from mindspore_federated._mindspore_federated import WorkerConfigItemPy_, WorkerRegisterItemPy_
from mindspore_federated.data_join.context import _WorkerRegister, _WorkerConfig


def worker_register_to_pybind_obj(worker_register: _WorkerRegister):
    """
    Parse a worker register from the pybind object.
    Inputs:
        worker_register (_WorkerRegister): the pybind object.
    """
    worker_register_item_py = WorkerRegisterItemPy_()
    worker_register_item_py.set_worker_name(worker_register.worker_name)
    return worker_register_item_py

def worker_config_to_pybind_obj(worker_config: _WorkerConfig):
    """
    Create a pybind object by the worker config.
    Inputs:
        worker_config (_WorkerConfig): the worker config object.
    """
    worker_config_item_py = WorkerConfigItemPy_()
    worker_config_item_py.set_primary_key(worker_config.primary_key)
    worker_config_item_py.set_bucket_num(worker_config.bucket_num)
    worker_config_item_py.set_shard_num(worker_config.shard_num)
    worker_config_item_py.set_join_type(worker_config.join_type)
    return worker_config_item_py


def pybind_obj_to_worker_config(worker_config_item_py: WorkerConfigItemPy_):
    """
    Parse a worker config from the pybind object.
    Inputs:
        workerConfigItemPy (WorkerConfigItemPy_): the pybind object.
    """
    primary_key = worker_config_item_py.primary_key()
    bucket_num = worker_config_item_py.bucket_num()
    shard_num = worker_config_item_py.shard_num()
    join_type = worker_config_item_py.join_type()
    logger.info("pybind_obj_to_worker_config, primary_key: {}".format(primary_key))
    logger.info("pybind_obj_to_worker_config, bucket_num: {}".format(bucket_num))
    logger.info("pybind_obj_to_worker_config, shard_num: {}".format(shard_num))
    logger.info("pybind_obj_to_worker_config, join_type: {}".format(join_type))
    return primary_key, bucket_num, shard_num, join_type
