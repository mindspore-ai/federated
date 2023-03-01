# pylint: disable=invalid-name

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
"""
MindSpore transformer
"""
from mindspore import __version__ as version
if version.startswith("2.0.0") and "a" not in version:
    from mindspore.parallel._transformer.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig
    from mindspore.parallel._transformer.loss import CrossEntropyLoss
    from mindspore.parallel._transformer import VocabEmbedding, TransformerEncoder, TransformerEncoderLayer, \
        AttentionMask
    from mindspore.parallel._transformer import MoEConfig
    from mindspore.parallel._transformer.layers import _LayerNorm, _Dropout
else:
    from mindspore.nn.transformer import TransformerOpParallelConfig, CrossEntropyLoss, TransformerRecomputeConfig
    from mindspore.nn.transformer import VocabEmbedding, TransformerEncoder, TransformerEncoderLayer, AttentionMask
    from mindspore.nn.transformer import MoEConfig
    from mindspore.nn.transformer.layers import _LayerNorm, _Dropout


TransformerOpParallelConfig = TransformerOpParallelConfig
TransformerRecomputeConfig = TransformerRecomputeConfig
CrossEntropyLoss = CrossEntropyLoss
VocabEmbedding = VocabEmbedding
TransformerEncoder = TransformerEncoder
TransformerEncoderLayer = TransformerEncoderLayer
AttentionMask = AttentionMask
MoEConfig = MoEConfig
_LayerNorm = _LayerNorm
_Dropout = _Dropout
