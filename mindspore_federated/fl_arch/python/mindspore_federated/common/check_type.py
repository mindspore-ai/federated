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
"""Essential tools to check the users input."""


def check_str(arg_name, str_val):
    """Check whether the input parameters are reasonable str input"""
    if not isinstance(str_val, str):
        raise RuntimeError(f"Parameter '{arg_name}' should be str, but actually {type(str_val)}")
    if not str_val:
        raise RuntimeError(f"Parameter '{arg_name}' should not be empty str")


def check_list(arg_name, list_val):
    """Check whether the input parameters are reasonable list input"""
    if not isinstance(list_val, list):
        raise RuntimeError(f"Parameter '{arg_name}' should be list, but actually {type(list)}")
    if not list_val:
        raise RuntimeError(f"Parameter '{arg_name}' should not be empty list")


def check_int(arg_name, int_val):
    """Check whether the input parameters are reasonable int input"""
    if not isinstance(int_val, int):
        raise RuntimeError(f"Parameter '{arg_name}' should be int, but actually {type(int_val)}")


def check_float(arg_name, float_val):
    """Check whether the input parameters are reasonable float input"""
    if not isinstance(float_val, float):
        raise RuntimeError(f"Parameter '{arg_name}' should be float, but actually {type(float_val)}")
