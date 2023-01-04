# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Compress config for vertical fl communication"""
from ..common import check_type

# pylint: disable=redefined-builtin
class CompressConfig:
    """
    Define the vertical server compress config.

    Args:
        ctype (str): Compress type for vertical fl communication
        quant_bits (int): Bits num of quant algorithm
    """

    def __init__(self, type, quant_bits):
        check_type.check_str("type", type)
        check_type.check_int("quant_bits", quant_bits)
        self.type = type
        self.quant_bits = quant_bits


def init_compress_config(compress_config):
    """
    Initialize vertical communication compress config.

    Args:
        compress_config (CompressConfig): Compress config of vertical communication.
    """
    if compress_config is not None:
        if not isinstance(compress_config, CompressConfig):
            raise RuntimeError(f"Parameter 'compress_config' should be instance of CompressConfig,"
                               f"but got {type(compress_config)}")
