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

"""Parse arguments"""

import ast
import argparse
import collections
import logging
import yaml


def parse_cmdline_with_yaml(parser, cfg, helper=None, choices=None, cfg_path="default_config.yaml"):
    """
    Parse command line arguments to the configuration according to the default yaml.

    Args:
        parser: Parent parser.
        cfg: Base configuration.
        helper: Helper description.
        cfg_path: Path to the default yaml config.
    """
    parser = argparse.ArgumentParser(description="[REPLACE THIS at config.py]",
                                     parents=[parser])
    helper = collections.OrderedDict() if helper is None else helper
    choices = collections.OrderedDict() if choices is None else choices
    for item in cfg:
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
            help_description = helper[item] if item in helper else "Please reference to {}".format(cfg_path)
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument("--" + item, type=ast.literal_eval, default=cfg[item], choices=choice,
                                    help=help_description)
            else:
                parser.add_argument("--" + item, type=type(cfg[item]), default=cfg[item], choices=choice,
                                    help=help_description)
    args = parser.parse_args()
    return args


def parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = yaml.load(fin.read(), Loader=yaml.FullLoader)
            def_vals = collections.OrderedDict()
            cfg_helper = collections.OrderedDict()
            cfg_choices = collections.OrderedDict()
            for key, value in cfgs.items():
                if not isinstance(value, dict):
                    raise ValueError("Config item must be dict contain {def_val, help , [choices]} ")
                if "def_val" in value:
                    def_vals[key] = value["def_val"]
                if "help" in value:
                    cfg_helper[key] = value["help"]
                if "choices" in value:
                    cfg_choices[key] = value["choices"]
        except:
            raise ValueError("Failed to parse yaml")
    return def_vals, cfg_helper, cfg_choices


def get_config(cfg_file):
    """
    Parse yaml file to get configuration information.

    Args:
        cfg_file(str):the directory of yaml file.

    Returns:
        argparse, the configuration information parsed from yaml file.

    Note:
        Using this function get configuration information to construct FLDataWorker.

    Examples:
        >>> current_dir = os.path.dirname(os.path.abspath(__file__))
        >>> args = get_config(os.path.join(current_dir, "vfl/vfl_data_join_config.yaml"))
        >>> dict_cfg = args.__dict__
        >>>
        >>> worker = FLDataWorker(config=dict_cfg)
        ...
    """
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    default, helper, choices = parse_yaml(cfg_file)
    args = parse_cmdline_with_yaml(parser=parser, cfg=default, helper=helper, choices=choices,
                                   cfg_path=cfg_file)
    for key in args.__dict__:
        logging.info("[%s]%s", key, args.__dict__[key])
    logging.info("Please check the above information for the configurations.")
    return args
