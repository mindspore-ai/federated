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

"""
Use to load yaml config file
"""
import os
import yaml
from mindspore_federated._mindspore_federated import YamlConfigItem_
from mindspore_federated._mindspore_federated import FLContext


def _load_yaml_config_file(yaml_config_file):
    """
    load yaml config file
    """
    if not os.path.exists(yaml_config_file):
        raise RuntimeError(f"yaml config file {yaml_config_file} not exist")
    with open(yaml_config_file, "r") as fp:
        config_content = fp.read()
    yaml_config = yaml.load(config_content, yaml.Loader)
    if not isinstance(yaml_config, dict):
        raise RuntimeError(f"Expect content of yaml config file {yaml_config_file} is of the dict type"
                           f", yaml config file: {yaml_config_file}")
    yaml_config_map = {}

    def load_yaml_dict(prefix, dict_config):
        for key, val in dict_config.items():
            config = YamlConfigItem_()
            if isinstance(val, bool):
                config.set_bool_val(val)
            elif isinstance(val, int):
                config.set_int_val(val)
            elif isinstance(val, float):
                config.set_float_val(val)
            elif isinstance(val, str):
                config.set_str_val(val)
            elif isinstance(val, dict):
                config.set_dict()
                load_yaml_dict(prefix + key + ".", val)
            else:
                continue
            yaml_config_map[prefix + key] = config

    load_yaml_dict("", yaml_config)
    return yaml_config_map


def load_yaml_config(yaml_config_file, role):
    """
    load yaml config
    """
    ctx = FLContext.get_instance()
    _load_yaml_config_file(yaml_config_file)
    yaml_config_map = _load_yaml_config_file(yaml_config_file)
    ctx.load_yaml_config(yaml_config_map, yaml_config_file, role)
