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
"""Essential tools to encoding the training communication compress."""


def quant_compress(data_list, quant_bits):
    """quant compression"""
    min_val = min(data_list)
    max_val = max(data_list)

    quant_p1 = ((1 << quant_bits) - 1) / (max_val - min_val + 1e-10)
    quant_p2 = int(round(quant_p1 * min_val))

    compress_data_list = []
    for data in data_list:
        compress_data = round(quant_p1 * data) - quant_p2
        compress_data_list.append(compress_data)
    return min_val, max_val, compress_data_list
