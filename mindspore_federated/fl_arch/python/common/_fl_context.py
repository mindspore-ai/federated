# Copyright 2020 Huawei Technologies Co., Ltd
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

import os
from mindspore_federated._mindspore_federated import FLContext
from mindspore_federated.python import log as logger
from mindspore.train.serialization import load_checkpoint, export

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
    "server_mode": fl_context().set_server_mode,
    "ms_role": fl_context().set_ms_role,
    "worker_num": fl_context().set_worker_num,
    "server_num": fl_context().set_server_num,
    "scheduler_ip": fl_context().set_scheduler_ip,
    "scheduler_port": fl_context().set_scheduler_port,
    "fl_server_port": fl_context().set_fl_server_port,
    "enable_fl_client": fl_context().set_fl_client_enable,
    "start_fl_job_threshold": fl_context().set_start_fl_job_threshold,
    "start_fl_job_time_window": fl_context().set_start_fl_job_time_window,
    "update_model_ratio": fl_context().set_update_model_ratio,
    "update_model_time_window": fl_context().set_update_model_time_window,
    "share_secrets_ratio": fl_context().set_share_secrets_ratio,
    "cipher_time_window": fl_context().set_cipher_time_window,
    "reconstruct_secrets_threshold": fl_context().set_reconstruct_secrets_threshold,
    "fl_name": fl_context().set_fl_name,
    "fl_iteration_num": fl_context().set_fl_iteration_num,
    "client_epoch_num": fl_context().set_client_epoch_num,
    "client_batch_size": fl_context().set_client_batch_size,
    "client_learning_rate": fl_context().set_client_learning_rate,
    "worker_step_num_per_iteration": fl_context().set_worker_step_num_per_iteration,
    "root_first_ca_path": fl_context().set_root_first_ca_path,
    "root_second_ca_path": fl_context().set_root_second_ca_path,
    "pki_verify": fl_context().set_pki_verify,
    "equip_crl_path": fl_context().set_equip_crl_path,
    "replay_attack_time_diff": fl_context().set_replay_attack_time_diff,
    "enable_ssl": fl_context().set_enable_ssl,
    "client_password": fl_context().set_client_password,
    "server_password": fl_context().set_server_password,
    "scheduler_manage_port": fl_context().set_scheduler_manage_port,
    "config_file_path": fl_context().set_config_file_path,
    "dp_eps": fl_context().set_dp_eps,
    "dp_delta": fl_context().set_dp_delta,
    "dp_norm_clip": fl_context().set_dp_norm_clip,
    "encrypt_type": fl_context().set_encrypt_type,
    "http_url_prefix": fl_context().set_http_url_prefix,
    "global_iteration_time_window": fl_context().set_global_iteration_time_window,
    "sign_k": fl_context().set_sign_k,
    "sign_eps": fl_context().set_sign_eps,
    "sign_thr_ratio": fl_context().set_sign_thr_ratio,
    "sign_global_lr": fl_context().set_sign_global_lr,
    "sign_dim_out": fl_context().set_sign_dim_out,
    "checkpoint_dir": fl_context().set_checkpoint_dir,
    "upload_compress_type": fl_context().set_upload_compress_type,
    "upload_sparse_rate": fl_context().set_upload_sparse_rate,
    "download_compress_type": fl_context().set_download_compress_type,
    "instance_name": fl_context().set_instance_name,
    "participation_time_level": fl_context().set_participation_time_level,
    "continuous_failure_times": fl_context().set_continuous_failure_times,
    "feature_maps": fl_context().set_feature_maps,
}

_get_fl_context_func_map = {
    "server_mode": fl_context().server_mode,
    "ms_role": fl_context().ms_role,
    "worker_num": fl_context().worker_num,
    "server_num": fl_context().server_num,
    "scheduler_ip": fl_context().scheduler_ip,
    "scheduler_port": fl_context().scheduler_port,
    "fl_server_port": fl_context().fl_server_port,
    "enable_fl_client": fl_context().fl_client_enable,
    "start_fl_job_threshold": fl_context().start_fl_job_threshold,
    "start_fl_job_time_window": fl_context().start_fl_job_time_window,
    "update_model_ratio": fl_context().update_model_ratio,
    "update_model_time_window": fl_context().update_model_time_window,
    "share_secrets_ratio": fl_context().share_secrets_ratio,
    "cipher_time_window": fl_context().cipher_time_window,
    "reconstruct_secrets_threshold": fl_context().reconstruct_secrets_threshold,
    "fl_name": fl_context().fl_name,
    "fl_iteration_num": fl_context().fl_iteration_num,
    "client_epoch_num": fl_context().client_epoch_num,
    "client_batch_size": fl_context().client_batch_size,
    "client_learning_rate": fl_context().client_learning_rate,
    "worker_step_num_per_iteration": fl_context().worker_step_num_per_iteration,
    "dp_eps": fl_context().dp_eps,
    "dp_delta": fl_context().dp_delta,
    "dp_norm_clip": fl_context().dp_norm_clip,
    "encrypt_type": fl_context().encrypt_type,
    "root_first_ca_path": fl_context().root_first_ca_path,
    "root_second_ca_path": fl_context().root_second_ca_path,
    "pki_verify": fl_context().pki_verify,
    "equip_crl_path": fl_context().equip_crl_path,
    "replay_attack_time_diff": fl_context().replay_attack_time_diff,
    "enable_ssl": fl_context().enable_ssl,
    "client_password": fl_context().client_password,
    "server_password": fl_context().server_password,
    "scheduler_manage_port": fl_context().scheduler_manage_port,
    "config_file_path": fl_context().config_file_path,
    "http_url_prefix": fl_context().http_url_prefix,
    "global_iteration_time_window": fl_context().global_iteration_time_window,
    "sign_k": fl_context().sign_k,
    "sign_eps": fl_context().sign_eps,
    "sign_thr_ratio": fl_context().sign_thr_ratio,
    "sign_global_lr": fl_context().sign_global_lr,
    "sign_dim_out": fl_context().sign_dim_out,
    "checkpoint_dir": fl_context().checkpoint_dir,
    "upload_compress_type": fl_context().upload_compress_type,
    "upload_sparse_rate": fl_context().upload_sparse_rate,
    "download_compress_type": fl_context().download_compress_type,
    "instance_name": fl_context().instance_name,
    "participation_time_level": fl_context().participation_time_level,
    "continuous_failure_times": fl_context().continuous_failure_times,
}

_check_positive_int_keys = ["server_num", "scheduler_port", "fl_server_port",
                            "start_fl_job_threshold", "start_fl_job_time_window", "update_model_time_window",
                            "fl_iteration_num", "client_epoch_num", "client_batch_size", "cipher_time_window",
                            "reconstruct_secrets_threshold"]

_check_non_negative_int_keys = ["worker_num"]

_check_positive_float_keys = ["update_model_ratio", "client_learning_rate"]

_check_port_keys = ["scheduler_port", "fl_server_port"]

_check_string_keys = {
    "upload_compress_type": ["NO_COMPRESS", "DIFF_SPARSE_QUANT"],
    "download_compress_type": ["NO_COMPRESS", "QUANT"],
}

def _set_fl_context(kwargs):
    """
    Set parameter server training mode context.

    Note:
        Some other environment variables should also be set for parameter server training mode.
        These environment variables are listed below:

        .. code-block::

            MS_SERVER_NUM  # Server number
            MS_WORKER_NUM  # Worker number
            MS_SCHED_HOST  # Scheduler IP address
            MS_SCHED_PORT  # Scheduler port
            MS_ROLE        # The role of this process:
                           # MS_SCHED represents the scheduler,
                           # MS_WORKER represents the worker,
                           # MS_PSERVER represents the Server


    Args:
        enable_ps (bool): Whether to enable parameter server training mode.
                          Only after enable_ps is set True, the environment variables will be effective.
                          Default: False.
        config_file_path (string): Configuration file path used by recovery. Default: ''.
        scheduler_manage_port (int): scheduler manage port used to scale out/in. Default: 11202.
        enable_ssl (bool): Set PS SSL mode enabled or disabled. Default: False.
        client_password (str): Password to decrypt the secret key stored in the client certificate. Default: ''.
        server_password (str): Password to decrypt the secret key stored in the server certificate. Default: ''.

    Raises:
        ValueError: If input key is not the attribute in parameter server training mode context.

    Examples:
        >>> context.set_fl_context(enable_ps=True, enable_ssl=True, client_password='123456', server_password='123456')
    """
    kwargs = _check_conflict_value(kwargs)
    for key, value in kwargs.items():
        if key not in _set_fl_context_func_map:
            raise ValueError("Set PS context keyword %s is not recognized!" % key)
        logger.info("FL context key is {}, value is {}".format(key, value))
        if key == "checkpoint_dir" and (kwargs["ms_role"] == "MS_SERVER" or kwargs["ms_role"] == "MS_WORKER"):
            checkpoint_dir = kwargs["checkpoint_dir"]
            if (os.path.exists(checkpoint_dir)):
                for checkpoint_file in os.listdir(checkpoint_dir):
                    logger.info("Checkpoint file is {}".format(checkpoint_file))
                    file_prefix = kwargs["fl_name"] + "_iteration"
                    if checkpoint_file.startswith(file_prefix):
                        checkpoint_file_path = os.path.join(checkpoint_dir, checkpoint_file)
                        param_dict = load_checkpoint(checkpoint_file_path)
                        logger.info("Param_dict is {}".format(param_dict))

            weight_fullnames = list()
            weight_datas = list()
            weight_shapes = list()
            weight_types = list()
            for param_name, param_value in param_dict.items():
                weight_fullnames.append(param_name)
                weight_np = param_value.asnumpy()
                weight_shapes.append(list(weight_np.shape))
                weight_datas.append(weight_np.reshape(-1).tolist())
                weight_types.append(str(weight_np.dtype).title())
                logger.info("Weight fullname is {}".format(param_name))
            set_func = _set_fl_context_func_map["feature_maps"]
            set_func(weight_fullnames, weight_datas, weight_shapes, weight_types)
        set_func = _set_fl_context_func_map[key]
        set_func(value)


def _get_fl_context(attr_key):
    """
    Get parameter server training mode context attribute value according to the key.

    Args:
        attr_key (str): The key of the attribute.

    Returns:
        Returns attribute value according to the key.

    Raises:
        ValueError: If input key is not attribute in auto parallel context.
    """
    if attr_key not in _get_fl_context_func_map:
        raise ValueError("Get PS context keyword %s is not recognized!" % attr_key)
    get_func = _get_fl_context_func_map[attr_key]
    value = get_func()
    return value


def _reset_fl_context():
    """
    Reset parameter server training mode context attributes to the default values:

    - enable_ps: False.
    """
    fl_context().reset()


def _is_role_worker():
    return fl_context().is_worker()


def _is_role_pserver():
    return fl_context().is_server()


def _is_role_sched():
    return fl_context().is_scheduler()


def _insert_hash_table_size(name, cache_vocab_size, embedding_size, vocab_size):
    fl_context().insert_hash_table_size(name, cache_vocab_size, embedding_size, vocab_size)


def _reinsert_hash_table_size(new_name, cur_name, cache_vocab_size, embedding_size):
    fl_context().reinsert_hash_table_size(new_name, cur_name, cache_vocab_size, embedding_size)


def _insert_weight_init_info(name, global_seed, op_seed):
    fl_context().insert_weight_init_info(name, global_seed, op_seed)


def _insert_accumu_init_info(name, init_val):
    fl_context().insert_accumu_init_info(name, init_val)


def _check_conflict_value(kwargs):
    if "upload_compress_type" in kwargs and "encrypt_type" in kwargs:
        if kwargs["upload_compress_type"] != "NO_COMPRESS" and kwargs["encrypt_type"] in ("SIGNDS", "PW_ENCRYPT"):
            logger.warning("The '{}' and '{}' are conflicted, and in '{}' mode the"
                           " 'upload_compress_type' will be 'NO_COMPRESS'".format(kwargs["encrypt_type"],
                                                                                  kwargs["upload_compress_type"],
                                                                                  kwargs["encrypt_type"]))
            kwargs["upload_compress_type"] = "NO_COMPRESS"
    return kwargs
