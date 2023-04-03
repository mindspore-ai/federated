# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore_federated.common import _checkparam as validator
from mindspore_federated._mindspore_federated import run_min_max_compress, run_min_max_decompress
from mindspore_federated._mindspore_federated import run_bit_pack, run_bit_unpack

COMPRESS_TYPE_FUNC_DICT = {
    "min_max": run_min_max_compress,
    "bit_pack": run_bit_pack,
}

DECOMPRESS_TYPE_FUNC_DICT = {
    "min_max": run_min_max_decompress,
    "bit_pack": run_bit_unpack,
}

COMPRESS_SUPPORT_NPTYPES = ("double", "float16", "float32", "float64")

NO_COMPRESS_TYPE = "no_compress"


class CompressConfig:
    """
    Define the vertical server compress config.

    Args:
        compress_type (str): Compress type for vertical fl communication. Supports ["min_max", "bit_pack"].

                             - min_max: The min max quantization compress method.

                             - bit_pack: The bit pack compress method.

        bit_num (int): Bits num of quant algorithm. The value range is within [1, 8]. Default: 8.
    """

    def __init__(self, compress_type, bit_num=8):
        validator.check_string(compress_type, COMPRESS_TYPE_FUNC_DICT.keys(), arg_name="compress_type")
        validator.check_int_range(bit_num, 1, 8, validator.INC_BOTH, arg_name="bit_num")
        self.compress_type = compress_type
        self.bit_num = bit_num
