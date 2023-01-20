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
"""Context for parameter server training mode"""

from mindspore_federated._mindspore_federated import FLContext
from .. import log as logger

_fl_context = None


def fl_context():
    """
    Get the global _fl_context, if it is not created, create a new one.

    Returns:
        _fl_context, the global parameter server training mode context.
    """
    global _fl_context
    if _fl_context is None:
        _fl_context = FLContext.get_instance()
    return _fl_context


_set_fl_context_func_map = {
}

_get_fl_context_func_map = {
}

_check_positive_int_keys = ["server_num", "scheduler_port",
                            "start_fl_job_threshold", "start_fl_job_time_window", "update_model_time_window",
                            "fl_iteration_num", "client_epoch_num", "client_batch_size", "cipher_time_window",
                            "reconstruct_secrets_threshold"]

_check_non_negative_int_keys = ["worker_num"]


def _check_conflict_value(kwargs):
    if "upload_compress_type" in kwargs and "encrypt_type" in kwargs:
        if kwargs["upload_compress_type"] != "NO_COMPRESS" and kwargs["encrypt_type"] in ("SIGNDS", "PW_ENCRYPT"):
            logger.warning("The '{}' and '{}' are conflicted, and in '{}' mode the"
                           " 'upload_compress_type' will be 'NO_COMPRESS'".format(kwargs["encrypt_type"],
                                                                                  kwargs["upload_compress_type"],
                                                                                  kwargs["encrypt_type"]))
            kwargs["upload_compress_type"] = "NO_COMPRESS"
    return kwargs


ROLE_OF_SERVER = "MS_SERVER"
ROLE_OF_WORKER = "MS_WORKER"
ROLE_OF_SCHEDULER = "MS_SCHED"

FEDAVG = "FedAvg"
FEDPROX = "FedProx"
SCAFFOLD = "Scaffold"
FEDNOVA = "FedNova"
SUPPORT_AGG_TYPES = (FEDAVG, FEDPROX, SCAFFOLD, FEDNOVA)

ENCRYPT_NONE = "NOT_ENCRYPT"
ENCRYPT_SIGNDS = "SIGNDS"
ENCRYPT_PW = "PW_ENCRYPT"
ENCRYPT_DP = "DP_ENCRYPT"
ENCRYPT_STABLE_PW = "STABLE_PW_ENCRYPT"
SUPPORT_ENC_TYPES_CLOUD = (ENCRYPT_NONE, ENCRYPT_STABLE_PW)

SERVER_MODE_FL = "FEDERATED_LEARNING"
SERVER_MODE_HYBRID = "HYBRID_TRAINING"
SERVER_MODE_CLOUD = "CLOUD_TRAINING"
