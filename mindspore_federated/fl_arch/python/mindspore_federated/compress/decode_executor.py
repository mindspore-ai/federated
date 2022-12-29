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
"""Essential tools to decoding the training communication compress."""


def quant_decompress(quant_compress_data, quant_bits, min_val, max_val):
    quant_p1 = ((1 << quant_bits) - 1) * 1.0 / (max_val - min_val + 1e-10)
    quant_p2 = int(round(quant_p1 * min_val))
    decompress_data_list = []
    for quant_data in quant_compress_data:
        decompress_data = (quant_data + quant_p2) / quant_p1
        decompress_data_list.append(decompress_data)
    return decompress_data_list
