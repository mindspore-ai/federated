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
"""Test the functions of FLYamlData"""
import os
from functools import wraps
import random
import yaml
import pytest

from mindspore_federated import FLYamlData
from mindspore_federated import log as logger


def yaml_test(func):
    """Decorator for test FLYamlData"""
    @wraps(func)
    def warp_test(*args, **kwargs):
        try:
            temp_dir = os.path.join(os.getcwd(), "temp")
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            func(*args, **kwargs)
        except Exception:
            logger.error("FLYamlData test catch exception")
            os.system(f"ls -l {temp_dir}/*.yaml && cat {temp_dir}/*.yaml")
            raise
        finally:
            os.system(f"rm -rf {temp_dir}")
    return warp_test


def test_fl_yaml_data_basic():
    """
    Feature: Parse and verify yaml file defining the vertical federated learning process.
    Description: Input the path of the yaml file, and return the corresponding parsed data structure.
    Expectation: Success
    """
    FLYamlData(os.path.join(os.getcwd(), 'yaml_files/leader.yaml'))
    FLYamlData(os.path.join(os.getcwd(), "yaml_files/follower.yaml"))


def test_fl_yaml_data_invalid_file():
    """
    Feature: Except ValueError when the path is invalid
    Description: Input an invalid file path.
    Expectation: Raise ValueError
    """
    with pytest.raises(ValueError):
        FLYamlData(os.path.join(os.getcwd(), "yaml_files/no.yaml"))


@yaml_test
def test_fl_yaml_data_incorrect_role():
    """
    Feature: Except ValueError when the field of 'role' is incorrect.
    Description: Generate a yaml with incorrect 'role' field, and try to parse it.
    Expectation: Raise ValueError
    """
    with open(os.path.join(os.getcwd(), 'yaml_files/leader.yaml'), 'r', encoding='utf-8') as fp:
        origin_yaml_data = yaml.safe_load(fp)
        fp.close()
        temp_dir = os.path.join(os.getcwd(), "temp")
        with open(os.path.join(temp_dir, 'incorrect_role.yaml'), encoding='utf-8', mode='w') as fp:
            origin_yaml_data['role'] = 'party'
            yaml.dump(data=origin_yaml_data, stream=fp, allow_unicode=True)
            fp.close()
            with pytest.raises(ValueError):
                FLYamlData(os.path.join(temp_dir, 'incorrect_role.yaml'))
        with open(os.path.join(temp_dir, 'no_role.yaml'), encoding='utf-8', mode='w') as fp:
            origin_yaml_data.pop('role')
            yaml.dump(data=origin_yaml_data, stream=fp, allow_unicode=True)
            fp.close()
            with pytest.raises(ValueError):
                FLYamlData(os.path.join(temp_dir, 'no_role.yaml'))


@yaml_test
def test_fl_yaml_data_no_opts():
    """
    Feature: Except ValueError when the field of 'opts' is empty.
    Description: Generate a yaml with empty 'opts' field, and try to parse it.
    Expectation: Raise ValueError
    """
    temp_dir = os.path.join(os.getcwd(), "temp")
    with open(os.path.join(os.getcwd(), 'yaml_files/leader.yaml'), 'r', encoding='utf-8') as fp:
        origin_yaml_data = yaml.safe_load(fp)
        fp.close()
        origin_yaml_data.pop('opts')
        with open(os.path.join(temp_dir, 'no_opts.yaml'), encoding='utf-8', mode='w') as fp_temp:
            yaml.dump(data=origin_yaml_data, stream=fp_temp, allow_unicode=True)
            fp_temp.close()
            with pytest.raises(ValueError):
                FLYamlData(os.path.join(temp_dir, 'no_opts.yaml'))


@yaml_test
def test_fl_yaml_data_no_model():
    """
    Feature: Except ValueError when the field of 'model' is empty.
    Description: Generate a yaml with empty 'model' field, and try to parse it.
    Expectation: Raise ValueError
    """
    temp_dir = os.path.join(os.getcwd(), "temp")
    with open(os.path.join(os.getcwd(), 'yaml_files/leader.yaml'), 'r', encoding='utf-8') as fp:
        origin_yaml_data = yaml.safe_load(fp)
        fp.close()
        origin_yaml_data.pop('opts')
        with open(os.path.join(temp_dir, 'no_model.yaml'), encoding='utf-8', mode='w') as fp_temp:
            yaml.dump(data=origin_yaml_data, stream=fp_temp, allow_unicode=True)
            fp_temp.close()
            with pytest.raises(ValueError):
                FLYamlData(os.path.join(temp_dir, 'no_model.yaml'))


@yaml_test
def test_fl_yaml_incorrect_grad_scalers():
    """
    Feature: Except ValueError when the input of grad_scalers is not contained in the network.
    Description: Generate a yaml with error 'grad_scalers' field, and try to parse it.
    Expectation: Raise ValueError
    """
    temp_dir = os.path.join(os.getcwd(), "temp")
    with open(os.path.join(os.getcwd(), 'yaml_files/leader.yaml'), 'r', encoding='utf-8') as fp:
        origin_yaml_data = yaml.safe_load(fp)
        fp.close()
        grad_scalers_yaml = origin_yaml_data['grad_scalers'][0]
        input_name_tmp = grad_scalers_yaml['inputs'][random.randint(0, 1)]['name']
        grad_scalers_yaml['inputs'][random.randint(0, 1)]['name'] = 'unknown'
        with open(os.path.join(temp_dir, 'incorrect_input_grad_scalers.yaml'), encoding='utf-8', mode='w') as fp_temp:
            yaml.dump(data=origin_yaml_data, stream=fp_temp, allow_unicode=True)
            fp_temp.close()
            with pytest.raises(ValueError):
                FLYamlData(os.path.join(temp_dir, 'incorrect_input_grad_scalers.yaml'))
        grad_scalers_yaml['inputs'][random.randint(0, 1)]['name'] = input_name_tmp
        grad_scalers_yaml.pop('output')
        with open(os.path.join(temp_dir, 'no_output_grad_scalers.yaml'), encoding='utf-8', mode='w') as fp_temp:
            yaml.dump(data=origin_yaml_data, stream=fp_temp, allow_unicode=True)
            fp_temp.close()
            with pytest.raises(ValueError):
                FLYamlData(os.path.join(temp_dir, 'no_output_grad_scalers.yaml'))


@yaml_test
def test_fl_yaml_incorrect_network():
    """
    Feature: Except ValueError when the input or output of networks is incorrect.
    Description: Generate a yaml with empty 'train_net' or 'eval_net' field, and try to parse it.
    Expectation: Raise ValueError
    """
    temp_dir = os.path.join(os.getcwd(), "temp")
    with open(os.path.join(os.getcwd(), 'yaml_files/leader.yaml'), 'r', encoding='utf-8') as fp:
        origin_yaml_data = yaml.safe_load(fp)
        fp.close()
        train_net = origin_yaml_data['model']['train_net']
        inputs_tmp = train_net['inputs']
        train_net.pop('inputs')
        with open(os.path.join(temp_dir, 'train_net_no_input.yaml'), encoding='utf-8', mode='w') as fp_temp:
            yaml.dump(data=origin_yaml_data, stream=fp_temp, allow_unicode=True)
            fp_temp.close()
            with pytest.raises(ValueError):
                FLYamlData(os.path.join(temp_dir, 'train_net_no_input.yaml'))
        train_net['inputs'] = inputs_tmp
        eval_net = origin_yaml_data['model']['eval_net']
        outputs_tmp = eval_net['outputs']
        eval_net.pop('outputs')
        with open(os.path.join(temp_dir, 'eval_net_no_output.yaml'), encoding='utf-8', mode='w') as fp_temp:
            yaml.dump(data=origin_yaml_data, stream=fp_temp, allow_unicode=True)
            fp_temp.close()
            with pytest.raises(ValueError):
                FLYamlData(os.path.join(temp_dir, 'eval_net_no_output.yaml'))
        eval_net['outputs'] = outputs_tmp
        eval_net['outputs'].clear()
        with open(os.path.join(temp_dir, 'eval_net_empty_output.yaml'), encoding='utf-8', mode='w') as fp_temp:
            yaml.dump(data=origin_yaml_data, stream=fp_temp, allow_unicode=True)
            fp_temp.close()
            with pytest.raises(ValueError):
                FLYamlData(os.path.join(temp_dir, 'eval_net_empty_output.yaml'))
