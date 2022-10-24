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
"""PanguAlpha model"""
import copy
import math
from sklearn.metrics import roc_auc_score

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn import Cell
from mindspore.nn.transformer.transformer import VocabEmbedding, TransformerEncoder, TransformerEncoderLayer, \
    AttentionMask
from mindspore.nn.transformer import MoEConfig
from mindspore.nn.transformer.layers import _LayerNorm, _Dropout
from mindspore.nn.metrics import Metric

class PPLMetric(Metric):
    """
    ppl metric
    """

    def __init__(self, data_length):
        super(PPLMetric, self).__init__()
        self.clear()
        self.data_length = data_length

    def clear(self):
        """Clear the internal evaluation result."""
        self.ppl = []
        self.tokens_count = 0

    def update(self, *inputs): # inputs
        """Update list of ppl"""
        logits = inputs[0].asnumpy().flatten().tolist() # logits
        self.ppl.append(logits[0] * self.data_length)
        self.tokens_count += 1

    def eval(self):
        val_loss = sum(self.ppl) / (self.tokens_count * self.data_length)
        ppl = math.exp(min(20, val_loss))
        return ppl

class AUCMetric(Metric):
    """
    Area under cure metric
    """

    def __init__(self):
        super(AUCMetric, self).__init__()
        self.true_labels = []
        self.pred_probs = []

    def clear(self):
        """Clear the internal evaluation result."""
        self.true_labels.clear()
        self.pred_probs.clear()

    def update(self, *inputs):
        """Update list of predicts and labels."""
        all_predict = inputs[1].asnumpy().flatten().tolist()
        all_label = inputs[2].asnumpy().flatten().tolist()
        self.pred_probs.extend(all_predict)
        self.true_labels.extend(all_label)

    def eval(self):
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError('true_labels.size is not equal to pred_probs.size()')
        return roc_auc_score(self.true_labels, self.pred_probs)


class EmbeddingLayer(nn.Cell):
    r"""Embedding layer of the PanGUAlpha Model"""

    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        # Only for the pipeline mode, the embedding needs to be row sliced.
        self.word_embedding = VocabEmbedding(vocab_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init=initializer("normal", [config.vocab_size, config.hidden_size],
                                                                    dtype=mstype.float32),
                                             parallel_config=config.parallel_config.embedding_dp_mp_config)
        copied_parallel_config = copy.deepcopy(config.parallel_config)
        copied_parallel_config.vocab_emb_dp = True
        self.position_embedding = VocabEmbedding(vocab_size=config.seq_length,
                                                 embedding_size=config.hidden_size,
                                                 param_init=initializer("normal",
                                                                        [config.seq_length, config.hidden_size],
                                                                        dtype=mstype.float32),
                                                 parallel_config=copied_parallel_config.embedding_dp_mp_config)
        self.add = P.Add().shard(
            ((config.parallel_config.data_parallel, 1, 1), (config.parallel_config.data_parallel, 1, 1)))
        self.dropout = _Dropout(1 - config.dropout_rate)
        self.dropout.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.batch_size = config.batch_size
        self.print = P.Print()

    def construct(self, input_ids, input_position, batch_valid_length):
        word_embedding, word_table = self.word_embedding(input_ids)
        if self.use_past and not self.is_first_iteration:
            _, seq_length = F.shape(input_ids)
            input_position = batch_valid_length.view(self.batch_size, seq_length)
        position_embedding, _ = self.position_embedding(input_position)
        embed = self.add(word_embedding, position_embedding)
        embed = self.dropout(embed)
        return embed, word_table

    def get_word_embedding_weight(self):
        return self.word_embedding.embedding_table


class QueryLayer(TransformerEncoderLayer):
    r"""Query Layer at the final layer."""

    def __init__(self, batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 param_init_type=mstype.float32,
                 hidden_act='fast_gelu',
                 use_past=False,
                 parallel_config=None,
                 softmax_compute_type=mstype.float32):
        super(QueryLayer, self).__init__(batch_size=batch_size,
                                         hidden_size=hidden_size,
                                         ffn_hidden_size=ffn_hidden_size,
                                         num_heads=num_heads,
                                         seq_length=seq_length,
                                         attention_dropout_rate=attention_dropout_rate,
                                         hidden_dropout_rate=hidden_dropout_rate,
                                         post_layernorm_residual=post_layernorm_residual,
                                         param_init_type=param_init_type,
                                         hidden_act=hidden_act,
                                         use_past=use_past,
                                         parallel_config=parallel_config.dp_mp_config,
                                         softmax_compute_type=softmax_compute_type)

    def construct(self, x, query_vector, input_mask, init_reset=True, batch_valid_length=None):
        r"""
        The forward process of the block.
        """
        # [bs * seq_length, embedding_size]
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            key_reset = self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
            value_reset = self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
            # add dependency for desired execution order
            input_x = F.depend(input_x, key_reset)
            input_x = F.depend(input_x, value_reset)

        attention, layer_present = self.attention(query_vector, input_x, input_x, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            key_update = self.assign(self.key_past, key_present)
            value_update = self.assign(self.value_past, value_present)
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)

        if self.post_layernorm_residual:
            output = self.add(output_x, mlp_logit)
        else:
            output = self.add(x, mlp_logit)
        return output, layer_present


class PanGuHead(Cell):
    """
    Head to get the logits of each token in the vocab
    Args:
        config(): the config of network
    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary
    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self, config):
        super(PanGuHead, self).__init__(config)
        self.matmul = P.MatMul(transpose_b=True).shard(((config.parallel_config.data_parallel, 1), (1, 1)))
        self.hidden_size = config.hidden_size
        self.dtype = mstype.float16
        self.cast = P.Cast()

    def construct(self, state, embed):
        state = P.Reshape()(state, (-1, self.hidden_size))
        # output logits over vocabulary [bs*seq_length, vocab_size]
        logits = self.matmul(self.cast(state, self.dtype), self.cast(embed, self.dtype))
        return logits


def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.


        Args:
            network(Cell) - Represents the transformer block
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
    """
    # Used for the pipeline's stages setting
    # As the final layer is not included here, so we need to manually add here.
    # original:  if set two stages, layers on two stages will be [15, 16+1]
    # with 1 added, the layers on two stages will be [16, 15 +1]
    pp_dis = max(int((layers + 1) / parallel_config.pipeline_stage), 1)
    # the pipeline stage must be in [0, parallel_config.pipeline_stage - 1]
    pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
    network.pipeline_stage = pp_id
    print(f"pipeline stage id is {pp_id}", flush=True)

    # Used for fusion tag of optimizer
    dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        # we give the fusion in pipeline mode a fixed value, otherwise the performance may become worse.
        network.set_comm_fusion(2)
    else:
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)


def reshape_to_2d(x):
    r"""reshape nd tensor to 2d, if n <= 2, keep original shape."""
    shape = F.shape(x)
    if len(shape) <= 2:
        return x
    x = F.reshape(x, (-1, shape[-1]))
    return x


class PanguAlphaModel(Cell):
    """
        The base backbone of the PanGuAlpha model
        Args:
            config(PanguAlphaConfig): the config of network
        Inputs:
            embedding_table(Tensor): embedding table which is the output of the embedding layer
            encoder_masks(Tensor): encoder mask which is the output of the embedding layer
            init_reset(bool): whether reset the init state of the network weights, default: True
            batch_valid_length(bool): whether the batch has valid length, default: None
        Returns:
            hidden_states(Tensor): the output embedding of the backbone
        """
    def __init__(self, config):
        super(PanguAlphaModel, self).__init__()
        self.is_pipeline = config.parallel_config.pipeline_stage > 1
        self.config = config

        self.num_layers = config.num_layers
        if config.use_moe:
            moe_config = MoEConfig(expert_num=config.expert_num,
                                   num_experts_chosen=config.per_token_num_experts_chosen)
        else:
            moe_config = MoEConfig(expert_num=1)
        # The shard setting of Transformer is set within the class StackedTransformer
        self.blocks = TransformerEncoder(num_layers=config.num_layers - 1,
                                         batch_size=config.batch_size,
                                         hidden_size=config.hidden_size,
                                         ffn_hidden_size=config.ffn_hidden_size,
                                         num_heads=config.num_heads,
                                         seq_length=config.seq_length,
                                         attention_dropout_rate=config.dropout_rate,
                                         hidden_dropout_rate=config.dropout_rate,
                                         lambda_func=set_parallel_configure_for_layer,
                                         hidden_act=config.hidden_act,
                                         param_init_type=config.param_init_type,
                                         use_past=config.use_past,
                                         parallel_config=config.parallel_config,
                                         moe_config=moe_config,
                                         softmax_compute_type=config.softmax_compute_type).blocks

        self.dtype = mstype.float16

        if config.load_ckpt_path:
            self.load_embedding_from_ckpt(config.load_ckpt_path)
        self.run_type = config.run_type

    def construct(self, embedding_table, encoder_masks, init_reset=True, batch_valid_length=None):
        r"""forward pass of the model"""
        hidden_states = P.Cast()(embedding_table, self.dtype)
        # the input of the incremental prediction is 3d
        if self.run_type != 'predict':
            hidden_states = reshape_to_2d(hidden_states)
        if self.blocks is not None:
            for i in range(self.num_layers - 1):
                hidden_states, _ = self.blocks[i](hidden_states, encoder_masks, init_reset, batch_valid_length)
        return hidden_states


class BackboneLossNet(nn.Cell):
    """
    Net of the backbone party, which is the 2nd sub-network and is deployed on the server B.
    Args:
        net (Cell): PanguAlphaModel, which is the 2nd sub-network.
    """

    def __init__(self, net: PanguAlphaModel):
        super(BackboneLossNet, self).__init__(auto_prefix=False)
        self.net = net

    def construct(self, embedding_table, word_table, position_id, attention_mask):
        hidden_states = self.net(embedding_table, attention_mask)
        return hidden_states, word_table, position_id, attention_mask


class BackboneNecessrayLossNet(nn.Cell):
    """
    Net of the backbone party, which is the 2nd sub-network and is deployed on the server B.
    Args:
        net (Cell): PanguAlphaModel, which is the 2nd sub-network.
    """

    def __init__(self, net: PanguAlphaModel):
        super(BackboneNecessrayLossNet, self).__init__(auto_prefix=False)
        self.net = net

    def construct(self, embedding_table, attention_mask):
        hidden_states = self.net(embedding_table, attention_mask)
        return hidden_states

class BackboneEvalNet(nn.Cell):
    """
    Eval net of the backbone party, which warps PanguAlphaModel.
    Args:
        backbone (class): PanguAlphaModel, which is the 2nd sub-network.
    """

    def __init__(self, backbone: PanguAlphaModel, generate=False, pad_token=6, seq_length=1024):
        super(BackboneEvalNet, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.pad_token = pad_token
        self.argmax = P.Argmax()
        self.generate = generate
        self.topk = P.TopK(sorted=True).shard(((1, 1),))
        self.gather = P.GatherV2().shard(((1, 1), (1,)))
        self.log_softmax = P.LogSoftmax().shard(((1, 1, 1),))
        self.get_attention_mask = AttentionMask(seq_length)
        self.expand = P.ExpandDims().shard(((1, 1, 1),))

    def construct(self, embedding_table, word_table, input_ids, current_index, init_reset=True,
                  batch_valid_length=None):
        """forward process of LeaderEvalNet"""
        input_mask = F.cast(F.not_equal(input_ids, self.pad_token), mstype.float32)
        bs, seq_length = F.shape(input_ids)
        attention_mask = self.get_attention_mask(input_mask)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (bs, 1))
        logits = self.backbone(embedding_table, word_table, input_position, attention_mask,
                               init_reset, batch_valid_length)
        index = current_index.view(1,)
        logits = self.gather(logits, index, 0)
        logits = logits.view(bs, 1, -1)
        log_probs = self.log_softmax(logits)
        return log_probs


class EmbeddingLossNet(nn.Cell):
    """
    Train net of the embedding party, or the tail sub-network.
    Args:
        net (class): EmbeddingLayer, which is the 1st sub-network.
        config (class): default config info.
    """

    def __init__(self, net: EmbeddingLayer, config):
        super(EmbeddingLossNet, self).__init__(auto_prefix=False)

        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        dp = config.parallel_config.data_parallel
        self.eod_token = config.eod_token
        self.net = net
        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.slice2 = P.StridedSlice().shard(((dp, 1, 1),))

    def construct(self, input_ids, position_id, attention_mask):
        """forward process of FollowerLossNet"""
        tokens = self.slice(input_ids, (0, 0), (self.batch_size, -1), (1, 1))
        embedding_table, word_table = self.net(tokens, position_id, batch_valid_length=None)
        return embedding_table, word_table, position_id, attention_mask

class EmbeddingNecessaryLossNet(nn.Cell):
    """
    Train net of the embedding party, or the tail sub-network.
    Args:
        net (class): EmbeddingLayer, which is the 1st sub-network.
        config (class): default config info.
    """

    def __init__(self, net: EmbeddingLayer, config):
        super(EmbeddingNecessaryLossNet, self).__init__(auto_prefix=False)

        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        dp = config.parallel_config.data_parallel
        self.eod_token = config.eod_token
        self.net = net
        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.slice2 = P.StridedSlice().shard(((dp, 1, 1),))

    def construct(self, input_ids, position_id, attention_mask):
        """forward process of FollowerLossNet"""
        tokens = self.slice(input_ids, (0, 0), (self.batch_size, -1), (1, 1))
        embedding_table, word_table = self.net(tokens, position_id, batch_valid_length=None)
        return embedding_table, word_table, attention_mask


class HeadLossNet(nn.Cell):
    """
    Train net of the head party, or the head sub-network.
    Args:
        net (class): PanGuHead, which is the 3rd sub-network.
        config (class): default config info.
    """

    def __init__(self, net: PanGuHead, loss, config):
        super(HeadLossNet, self).__init__(auto_prefix=False)
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        dp = config.parallel_config.data_parallel
        self.network = net
        self.eod_token = config.eod_token
        self.loss = loss
        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.top_query_layer = QueryLayer(batch_size=config.batch_size,
                                          hidden_size=config.hidden_size,
                                          ffn_hidden_size=config.ffn_hidden_size,
                                          num_heads=config.num_heads,
                                          seq_length=config.seq_length,
                                          attention_dropout_rate=config.dropout_rate,
                                          hidden_dropout_rate=config.dropout_rate,
                                          hidden_act=config.hidden_act,
                                          param_init_type=config.param_init_type,
                                          use_past=config.use_past,
                                          parallel_config=config.parallel_config)
        self.top_query_layer.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.dtype = mstype.float16
        self.layernorm = _LayerNorm((config.hidden_size,)).to_float(mstype.float32)
        if config.parallel_config.pipeline_stage > 1:
            self.layernorm.set_comm_fusion(2)
        else:
            self.layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.layernorm.shard(((config.parallel_config.data_parallel, 1),))

        copied_parallel_config = copy.deepcopy(config.parallel_config)
        copied_parallel_config.vocab_emb_dp = True
        self.top_query_embedding = VocabEmbedding(vocab_size=config.seq_length,
                                                  embedding_size=config.hidden_size,
                                                  param_init=initializer("normal",
                                                                         [config.seq_length, config.hidden_size],
                                                                         dtype=mstype.float32),
                                                  parallel_config=copied_parallel_config.embedding_dp_mp_config)
        if config.parallel_config.pipeline_stage > 1:
            self.top_query_embedding.set_comm_fusion(2)
        else:
            self.top_query_embedding.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

    def construct(self, hidden_states, input_ids, word_table, position_id, attention_mask):
        r"""Forward process of the pangu alpha model"""
        hidden_states = reshape_to_2d(hidden_states)
        encoder_output = self.layernorm(hidden_states)
        encoder_output = P.Cast()(encoder_output, self.dtype)
        top_query_hidden_states, _ = self.top_query_embedding(position_id)
        top_query_hidden_states = reshape_to_2d(top_query_hidden_states)
        encoder_output, _ = self.top_query_layer(encoder_output, top_query_hidden_states,
                                                 attention_mask, True, batch_valid_length=None)
        tokens = self.slice(input_ids, (0, 0), (self.batch_size, -1), (1, 1))
        input_mask = F.cast(self.not_equal(tokens, self.eod_token),
                            mstype.float32)
        logits = self.network(encoder_output, word_table)
        # Get label corresponding to input tokens
        labels = self.slice(input_ids, (0, 1), (self.batch_size, self.len + 1), (1, 1))
        labels = P.Reshape()(labels, (-1,))
        input_mask = P.Reshape()(input_mask, (-1,))
        output = self.loss(logits, labels, input_mask)
        return output
