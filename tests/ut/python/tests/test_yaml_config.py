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
"""Test the functions of server in server mode FEDERATED_LEARNING"""

import numpy as np

from common import fl_name_with_idx, make_yaml_config, start_fl_server, fl_test

from mindspore_federated import FeatureMap

start_fl_job_reach_threshold_rsp = "Current amount for startFLJob has reached the threshold"
update_model_reach_threshold_rsp = "Current amount for updateModel is enough."


def create_default_feature_map():
    update_feature_map = {"feature_conv": np.random.randn(2, 3).astype(np.float32),
                          "feature_bn": np.random.randn(1).astype(np.float32),
                          "feature_bn2": np.random.randn(1).astype(np.float32).reshape(tuple()),  # scalar
                          "feature_conv2": np.random.randn(2, 3).astype(np.float32)}
    return update_feature_map


# pylint: disable=R1710
def val_type_str(val):
    if isinstance(val, str):
        return "str"
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    assert False


@fl_test
def test_yaml_config_multi_server_pki_verify_not_match_failed():
    """
    Feature: Yaml config
    Description: hyper params of pki_verify != value of first server
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, start_fl_job_threshold=2)

    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)

    http_server_address2 = "127.0.0.1:3002"
    try:
        yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
        make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, start_fl_job_threshold=2, pki_verify=True)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address2)
        assert False
    except RuntimeError as e:
        assert "Sync hyper params with distributed cache failed" in str(e)


@fl_test
def test_yaml_config_invalid_round_config_val_type_failed():
    """
    Feature: Yaml config
    Description: Yaml config round config val type invalid
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    check_map = {
        "round.start_fl_job_threshold": 1,
        "round.start_fl_job_time_window": 10000,
        "round.update_model_ratio": 1.0,
        "round.update_model_time_window": 10000,
        "round.global_iteration_time_window": 30000,
    }
    for key, val in check_map.items():
        try:
            make_yaml_config(fl_name, {f"{key}": f"{val}"},
                             output_yaml_file=yaml_config_file)
            start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                            http_server_address=http_server_address)
            assert False
        except RuntimeError as e:
            assert f"The parameter '{key}' is expected to be type {val_type_str(val)}" in str(e)

    # expect int, got dict
    try:
        make_yaml_config(fl_name, {"round.global_iteration_time_window": {"key": 3000.0}},
                         output_yaml_file=yaml_config_file)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert "The parameter 'round.global_iteration_time_window' is expected to be type int" in str(e)


@fl_test
def test_yaml_config_invalid_round_config_val_range_failed():
    """
    Feature: Yaml config
    Description: Yaml config round config val range invalid
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    invalid_vals = {
        "round.start_fl_job_threshold": [0, 0x100000000],  # [1, UINT32_MAX]
        "round.start_fl_job_time_window": [0, 0x100000000],  # [1, UINT32_MAX]
        "round.update_model_ratio": [0, 1.01],  # (0, 1.0]
        "round.update_model_time_window": [0, 0x100000000],  # [1, UINT32_MAX]
        "round.global_iteration_time_window": [0, 0x100000000],  # [1, UINT32_MAX]
    }
    for key, val in invalid_vals.items():
        for invalid_val in val:
            try:
                make_yaml_config(fl_name, {key: invalid_val}, output_yaml_file=yaml_config_file)
                start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                                http_server_address=http_server_address)
                assert False
            except RuntimeError as e:
                assert f"Failed to check value of parameter '{key}'" in str(e)


@fl_test
def test_yaml_config_missing_round_config_failed():
    """
    Feature: Yaml config
    Description: Yaml config round config val type invalid
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    # round is missing
    try:
        make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, rmv_configs=["round"])
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert "The parameter 'round.start_fl_job_threshold' is missing" in str(e)

    require_params = ["round.start_fl_job_threshold", "round.start_fl_job_time_window", "round.update_model_ratio",
                      "round.update_model_time_window", "round.global_iteration_time_window"]

    for item in require_params:
        try:
            make_yaml_config(fl_name, {}, output_yaml_file=yaml_config_file, rmv_configs=[item])
            start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                            http_server_address=http_server_address)
            assert False
        except RuntimeError as e:
            assert f"The parameter '{item}' is missing" in str(e)


@fl_test
def test_yaml_config_invalid_encrypt_config_encrypt_type_failed():
    """
    Feature: Yaml config
    Description: Yaml config round config val type invalid
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    # expect str, got int
    try:
        make_yaml_config(fl_name, {"encrypt.encrypt_train_type": 0}, output_yaml_file=yaml_config_file)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert "The parameter 'encrypt.encrypt_train_type' is expected to be type str" in str(e)

    # encrypt.encrypt_train_type: NOT_ENCRYPT, PW_ENCRYPT, STABLE_PW_ENCRYPT, DP_ENCRYPT, SIGNDS
    try:
        make_yaml_config(fl_name, {"encrypt.encrypt_train_type": "ENCRYPT_INVALID"}, output_yaml_file=yaml_config_file)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert "The value of parameter 'encrypt.encrypt_train_type' can be only one of" in str(e)


@fl_test
def test_yaml_config_encrypt_config_pwe_success():
    """
    Feature: Yaml config encrypt PWE
    Description: Yaml config round config PWE
    Expectation: FL server start successfully
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    # minimum_secret_shares_for_reconstruct == 2 >= clients_threshold_for_reconstruct: reconstruct_secrets_threshold+1
    make_yaml_config(fl_name, {"encrypt.encrypt_train_type": "PW_ENCRYPT"}, start_fl_job_threshold=2,
                     output_yaml_file=yaml_config_file)
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)


@fl_test
def test_yaml_config_invalid_encrypt_config_pwe_val_type_failed():
    """
    Feature: Yaml config encrypt PWE
    Description: Yaml config round config pwe val type invalid
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    check_map = {
        "encrypt.pw_encrypt.share_secrets_ratio": 1.0,
        "encrypt.pw_encrypt.cipher_time_window": 3000,
        "encrypt.pw_encrypt.reconstruct_secrets_threshold": 1
    }
    for key, val in check_map.items():
        try:
            make_yaml_config(fl_name,
                             {"encrypt.encrypt_train_type": "PW_ENCRYPT", f"{key}": f"{val}"},
                             output_yaml_file=yaml_config_file)
            start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                            http_server_address=http_server_address)
            assert False
        except RuntimeError as e:
            assert f"The parameter '{key}' is expected to be type {val_type_str(val)}" in str(e)


@fl_test
def test_yaml_config_invalid_encrypt_config_pwe_val_range_failed():
    """
    Feature: Yaml config encrypt PWE
    Description: Yaml config pwe encrypt configs value range invalid
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    invalid_vals = {
        "encrypt.pw_encrypt.share_secrets_ratio": [0, 1.01],  # (0,1.0]
        "encrypt.pw_encrypt.cipher_time_window": [0, 0x100000000],  # [1, UINT32_MAX]
        "encrypt.pw_encrypt.reconstruct_secrets_threshold": [0, 0x100000000],  # [1, UINT32_MAX]
    }
    for key, val in invalid_vals.items():
        for invalid_val in val:
            try:
                make_yaml_config(fl_name, {"encrypt.encrypt_train_type": "PW_ENCRYPT", key: invalid_val},
                                 output_yaml_file=yaml_config_file)
                start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                                http_server_address=http_server_address)
                assert False
            except RuntimeError as e:
                assert f"Failed to check value of parameter '{key}'" in str(e)
    try:
        # minimum_secret_shares_for_reconstruct >= clients_threshold_for_reconstruct: reconstruct_secrets_threshold+1
        make_yaml_config(fl_name, {"encrypt.encrypt_train_type": "PW_ENCRYPT"}, start_fl_job_threshold=1,
                         output_yaml_file=yaml_config_file)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert "cipher init fail." in str(e)


@fl_test
def test_yaml_config_missing_encrypt_config_pwe_failed():
    """
    Feature: Yaml config encrypt PWE
    Description: Yaml config pwe encrypt configs are missing
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    # encrypt.pw_encrypt is empty
    try:
        make_yaml_config(fl_name, {"encrypt.encrypt_train_type": "PW_ENCRYPT"}, output_yaml_file=yaml_config_file,
                         rmv_configs=["encrypt.pw_encrypt"])
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert "The parameter 'encrypt.pw_encrypt.share_secrets_ratio' is missing" in str(e)

    require_params = ["encrypt.pw_encrypt.share_secrets_ratio",
                      "encrypt.pw_encrypt.cipher_time_window",
                      "encrypt.pw_encrypt.reconstruct_secrets_threshold"]
    for item in require_params:
        try:
            make_yaml_config(fl_name, {"encrypt.encrypt_train_type": "PW_ENCRYPT"}, output_yaml_file=yaml_config_file,
                             rmv_configs=[item])
            start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                            http_server_address=http_server_address)
            assert False
        except RuntimeError as e:
            assert f"The parameter '{item}' is missing" in str(e)


@fl_test
def test_yaml_config_encrypt_config_dp_success():
    """
    Feature: Yaml config encrypt DP
    Description: Yaml config round config DP
    Expectation: FL server start successfully
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    make_yaml_config(fl_name, {"encrypt.encrypt_train_type": "DP_ENCRYPT"}, output_yaml_file=yaml_config_file)
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)


@fl_test
def test_yaml_config_invalid_encrypt_config_dp_val_type_failed():
    """
    Feature: Yaml config encrypt DP
    Description: Yaml config round config DP val type invalid
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    check_map = {
        "encrypt.dp_encrypt.dp_eps": 50.0,
        "encrypt.dp_encrypt.dp_delta": 0.01,
        "encrypt.dp_encrypt.dp_norm_clip": 1.0,
    }
    for key, val in check_map.items():
        try:
            make_yaml_config(fl_name,
                             {"encrypt.encrypt_train_type": "DP_ENCRYPT", f"{key}": f"{val}"},
                             output_yaml_file=yaml_config_file)
            start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                            http_server_address=http_server_address)
            assert False
        except RuntimeError as e:
            assert f"The parameter '{key}' is expected to be type {val_type_str(val)}" in str(e)


@fl_test
def test_yaml_config_invalid_encrypt_config_dp_val_range_failed():
    """
    Feature: Yaml config
    Description: Yaml config dp encrypt configs value invalid
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    invalid_vals = {
        "encrypt.dp_encrypt.dp_eps": [0.0],  # > 0
        "encrypt.dp_encrypt.dp_delta": [0.0, 1.0],  # (0,1)
        "encrypt.dp_encrypt.dp_norm_clip": [0.0],  # >0
    }
    for key, val in invalid_vals.items():
        for invalid_val in val:
            try:
                make_yaml_config(fl_name, {"encrypt.encrypt_train_type": "DP_ENCRYPT", key: invalid_val},
                                 output_yaml_file=yaml_config_file)
                start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                                http_server_address=http_server_address)
                assert False
            except RuntimeError as e:
                assert f"Failed to check value of parameter '{key}'" in str(e)


@fl_test
def test_yaml_config_encrypt_config_signds_success():
    """
    Feature: Yaml config encrypt SIGNDS
    Description: Yaml config round config SIGNDS
    Expectation: FL server start successfully
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    make_yaml_config(fl_name, {"encrypt.encrypt_train_type": "SIGNDS"}, output_yaml_file=yaml_config_file)
    start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file, http_server_address=http_server_address)


@fl_test
def test_yaml_config_invalid_encrypt_config_signds_val_type_failed():
    """
    Feature: Yaml config encrypt SignDs
    Description: Yaml config round config SignDs val type invalid
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    check_map = {
        "encrypt.signds.sign_k": 0.01,
        "encrypt.signds.sign_eps": 100.0,
        "encrypt.signds.sign_thr_ratio": 0.6,
        "encrypt.signds.sign_global_lr": 0.1,
        "encrypt.signds.sign_dim_out": 0,
    }
    for key, val in check_map.items():
        try:
            make_yaml_config(fl_name,
                             {"encrypt.encrypt_train_type": "SIGNDS", f"{key}": f"{val}"},
                             output_yaml_file=yaml_config_file)
            start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                            http_server_address=http_server_address)
            assert False
        except RuntimeError as e:
            assert f"The parameter '{key}' is expected to be type {val_type_str(val)}" in str(e)


@fl_test
def test_yaml_config_invalid_encrypt_config_signds_val_range_failed():
    """
    Feature: Yaml config
    Description: Yaml config dp encrypt configs value invalid
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    invalid_vals = {
        "encrypt.signds.sign_k": [0.0, 0.26],  # (0, 0.25]
        "encrypt.signds.sign_eps": [0, 101],  # (0, 100]
        "encrypt.signds.sign_thr_ratio": [0.49, 1.01],  # [0.5, 1]
        "encrypt.signds.sign_global_lr": [0.0],  # >0
        "encrypt.signds.sign_dim_out": [-1, 51]  # [0, 50]
    }
    for key, val in invalid_vals.items():
        for invalid_val in val:
            try:
                make_yaml_config(fl_name, {"encrypt.encrypt_train_type": "SIGNDS", key: invalid_val},
                                 output_yaml_file=yaml_config_file)
                start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                                http_server_address=http_server_address)
                assert False
            except RuntimeError as e:
                assert f"Failed to check value of parameter '{key}'" in str(e)


@fl_test
def test_yaml_config_invalid_compression_config_val_type_failed():
    """
    Feature: Yaml config encrypt SignDs
    Description: Yaml config compression config val type invalid
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    try:
        make_yaml_config(fl_name, {"compression.upload_compress_type": 1234}, output_yaml_file=yaml_config_file)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                        http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert f"The parameter 'compression.upload_compress_type' is expected to be type str" in str(e)
    try:
        make_yaml_config(fl_name, {"compression.upload_sparse_rate": "0.4"}, output_yaml_file=yaml_config_file)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                        http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert f"The parameter 'compression.upload_sparse_rate' is expected to be type float" in str(e)
    try:
        make_yaml_config(fl_name, {"compression.download_compress_type": 1234}, output_yaml_file=yaml_config_file)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                        http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert f"The parameter 'compression.download_compress_type' is expected to be type str" in str(e)


@fl_test
def test_yaml_config_invalid_compression_config_val_range_failed():
    """
    Feature: Yaml config compression
    Description: Yaml config dp compression configs value invalid
    Expectation: Exception will be raised
    """
    fl_name = fl_name_with_idx("FlTest")
    http_server_address = "127.0.0.1:3001"
    yaml_config_file = f"temp/yaml_{fl_name}_config.yaml"
    np.random.seed(0)
    feature_map = FeatureMap()
    init_feature_map = create_default_feature_map()
    feature_map.add_feature("feature_conv", init_feature_map["feature_conv"], require_aggr=True)
    feature_map.add_feature("feature_bn", init_feature_map["feature_bn"], require_aggr=True)
    feature_map.add_feature("feature_bn2", init_feature_map["feature_bn2"], require_aggr=True)
    feature_map.add_feature("feature_conv2", init_feature_map["feature_conv2"], require_aggr=False)

    # NO_COMPRESS, DIFF_SPARSE_QUANT
    try:
        make_yaml_config(fl_name, {"compression.upload_compress_type": "INVALID_QUANT"},
                         output_yaml_file=yaml_config_file)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                        http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert "The value of parameter 'compression.upload_compress_type' can be only one of" in str(e)

    # NO_COMPRESS, DIFF_SPARSE_QUANT
    try:
        make_yaml_config(fl_name, {"compression.upload_compress_type": "QUANT"},
                         output_yaml_file=yaml_config_file)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                        http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert "The value of parameter 'compression.upload_compress_type' can be only one of" in str(e)

    # NO_COMPRESS, QUANT
    try:
        make_yaml_config(fl_name, {"compression.download_compress_type": "DIFF_SPARSE_QUANT"},
                         output_yaml_file=yaml_config_file)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                        http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert "The value of parameter 'compression.download_compress_type' can be only one of" in str(e)

    # (0,1.0]
    try:
        make_yaml_config(fl_name, {"compression.upload_sparse_rate": 0},
                         output_yaml_file=yaml_config_file)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                        http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert "Failed to check value of parameter 'compression.upload_sparse_rate'" in str(e)

    # (0,1.0]
    try:
        make_yaml_config(fl_name, {"compression.upload_sparse_rate": 1.01},
                         output_yaml_file=yaml_config_file)
        start_fl_server(feature_map=feature_map, yaml_config=yaml_config_file,
                        http_server_address=http_server_address)
        assert False
    except RuntimeError as e:
        assert "Failed to check value of parameter 'compression.upload_sparse_rate'" in str(e)
