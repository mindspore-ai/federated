import os
import yaml
from mindspore_federated._mindspore_federated import YamlConfigItem_
from mindspore_federated._mindspore_federated import FLContext


def _load_yaml_config_file(yaml_config_file):
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


def load_yaml_config(yaml_config_file, role, enable_ssl):
    ctx = FLContext.get_instance()
    _load_yaml_config_file(yaml_config_file)
    yaml_config_map = _load_yaml_config_file(yaml_config_file)
    ctx.load_yaml_config(yaml_config_map, yaml_config_file, role, enable_ssl)
