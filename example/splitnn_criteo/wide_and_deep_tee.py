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
"""This module realize the wide and deep network with TEE for splitnn test on criteo dataset"""

from collections import OrderedDict

from mindspore import nn, context, ops
from mindspore import ParameterTuple, Tensor
import mindspore.common.dtype as mstype
from mindspore.nn import Dropout
from mindspore.nn.optim import Adam, FTRL, SGD
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from wide_and_deep import init_method, init_var_dict
from mindspore_federated.common import vfl_utils
from mindspore_federated._mindspore_federated import init_tee_cut_layer, backward_tee_cut_layer, \
    encrypt_client_data, secure_forward_tee_cut_layer

class DenseLayer(nn.Cell):
    """
    Dense Layer for Deep Layer of WideDeep Model;
    Containing: activation, matmul, bias_add;
    Args:
        input_dim (int): input dimension.
        output_dim (int): output dimension.
        weight_bias_init (list): str list indicating init methods for weights and biases.
        act_str (str): the type of activation function, including relu, sigmoid, and tanh.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1.
            E.g. rate=0.9, dropping out 10% of input units. Default: 0.5.
        use_activation (bool): whether use activation function. Default: True
        convert_dtype (bool): whether convert type of output value to float32. Default: True
        drop_out (bool): whether use dropout layer. Default: False
    """

    def __init__(self, input_dim, output_dim, weight_bias_init, act_str,
                 keep_prob=0.5, use_activation=True, convert_dtype=True, drop_out=False):
        super(DenseLayer, self).__init__()
        weight_init, bias_init = weight_bias_init
        self.weight = init_method(
            weight_init, [input_dim, output_dim], name="weight")
        self.bias = init_method(bias_init, [output_dim], name="bias")
        self.act_func = self._init_activation(act_str)
        self.matmul = ops.MatMul(transpose_b=False)
        self.bias_add = ops.BiasAdd()
        self.cast = ops.Cast()
        self.dropout = Dropout(keep_prob=keep_prob)
        self.use_activation = use_activation
        self.convert_dtype = convert_dtype
        self.drop_out = drop_out

    def _init_activation(self, act_str):
        act_str = act_str.lower()
        if act_str == "relu":
            act_func = ops.ReLU()
        elif act_str == "sigmoid":
            act_func = ops.Sigmoid()
        elif act_str == "tanh":
            act_func = ops.Tanh()
        return act_func

    def construct(self, x):
        """
        Construct Dense layer
        """
        if self.training and self.drop_out:
            x = self.dropout(x)
        if self.convert_dtype:
            weight = self.weight
            bias = self.bias
            wx = self.matmul(x, weight)
            wx = self.bias_add(wx, bias)
            if self.use_activation:
                wx = self.act_func(wx)
            wx = self.cast(wx, mstype.float32)
        else:
            wx = self.matmul(x, self.weight)
            wx = self.bias_add(wx, self.bias)
            if self.use_activation:
                wx = self.act_func(wx)
        return wx


class WideDeepModel(nn.Cell):
    """
    From paper: " Wide & Deep Learning for Recommender Systems". Input pre-processed data, and
        output wide and deep features.
    Args:
        config (Class): The default config of Wide&Deep.
        field_size (uint32): the feature size of input tensor.
    """

    def __init__(self, config, field_size):
        super(WideDeepModel, self).__init__()
        self.config = config
        self.batch_size = config.batch_size
        host_device_mix = bool(config.host_device_mix)
        parameter_server = bool(config.parameter_server)
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        is_auto_parallel = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        if is_auto_parallel:
            self.batch_size = self.batch_size * get_group_size()
            print('bs modified to %d' % self.batch_size)
        sparse = config.sparse
        self.field_size = field_size
        self.emb_dim = config.emb_dim
        self.weight_init, self.bias_init = config.weight_bias_init
        self.deep_input_dims = self.field_size * self.emb_dim
        self.all_dim_list = [self.deep_input_dims] + config.deep_layer_dim + [1]
        init_acts = [('Wide_b', [1], config.emb_init)]
        var_map = init_var_dict(config.init_args, init_acts)
        self.wide_b = var_map["Wide_b"]
        self.dense_layer_1 = DenseLayer(self.all_dim_list[0], self.all_dim_list[1],
                                        config.weight_bias_init, config.deep_layer_act,
                                        convert_dtype=True, drop_out=config.dropout_flag)
        self.dense_layer_2 = DenseLayer(self.all_dim_list[1], self.all_dim_list[2],
                                        config.weight_bias_init, config.deep_layer_act,
                                        convert_dtype=True, drop_out=config.dropout_flag)
        self.dense_layer_3 = DenseLayer(self.all_dim_list[2], self.all_dim_list[3],
                                        config.weight_bias_init, config.deep_layer_act,
                                        convert_dtype=True, drop_out=config.dropout_flag)
        self.dense_layer_4 = DenseLayer(self.all_dim_list[3], 1,
                                        config.weight_bias_init, config.deep_layer_act,
                                        convert_dtype=True, drop_out=config.dropout_flag)
        self.wide_mul = ops.Mul()
        self.deep_mul = ops.Mul()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.reshape = ops.Reshape()
        self.deep_reshape = ops.Reshape()
        self.concat = ops.Concat(axis=1)
        self._init_embedding(config, is_auto_parallel, sparse, host_device_mix, parameter_server)

    def _init_embedding(self, config, is_auto_parallel, sparse, host_device_mix, parameter_server):
        """init the embedding layer of the model"""
        if is_auto_parallel and sparse and not config.field_slice and not parameter_server:
            target = 'CPU' if host_device_mix else 'DEVICE'
            self.wide_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, 1, target=target,
                                                           slice_mode=nn.EmbeddingLookup.TABLE_ROW_SLICE)
            if config.deep_table_slice_mode == "column_slice":
                self.deep_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, self.emb_dim, target=target,
                                                               slice_mode=nn.EmbeddingLookup.TABLE_COLUMN_SLICE)
                self.dense_layer_1.dropout.dropout.shard(((1, get_group_size()),))
                self.dense_layer_1.matmul.shard(((1, get_group_size()), (get_group_size(), 1)))
                self.dense_layer_1.matmul.add_prim_attr("field_size", self.field_size)
                self.deep_mul.shard(((1, 1, get_group_size()), (1, 1, 1)))
                self.deep_reshape.add_prim_attr("skip_redistribution", True)
            else:
                self.deep_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, self.emb_dim, target=target,
                                                               slice_mode=nn.EmbeddingLookup.TABLE_ROW_SLICE)
            self.reduce_sum.add_prim_attr("cross_batch", True)
        elif is_auto_parallel and host_device_mix and config.field_slice and config.full_batch and config.manual_shape:
            manual_shapes = tuple((s[0] for s in config.manual_shape))
            self.deep_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, self.emb_dim,
                                                           slice_mode=nn.EmbeddingLookup.FIELD_SLICE,
                                                           manual_shapes=manual_shapes)
            self.wide_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, 1,
                                                           slice_mode=nn.EmbeddingLookup.FIELD_SLICE,
                                                           manual_shapes=manual_shapes)
            self.deep_mul.shard(((1, get_group_size(), 1), (1, get_group_size(), 1)))
            self.wide_mul.shard(((1, get_group_size(), 1), (1, get_group_size(), 1)))
            self.reduce_sum.shard(((1, get_group_size(), 1),))
            self.dense_layer_1.dropout.dropout.shard(((1, get_group_size()),))
            self.dense_layer_1.matmul.shard(((1, get_group_size()), (get_group_size(), 1)))
        elif parameter_server:
            cache_enable = config.vocab_cache_size > 0
            target = 'DEVICE' if cache_enable else 'CPU'
            if not cache_enable:
                sparse = True
            if is_auto_parallel and config.full_batch and cache_enable:
                self.deep_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, self.emb_dim, target=target,
                                                               slice_mode=nn.EmbeddingLookup.TABLE_ROW_SLICE,
                                                               sparse=sparse,
                                                               vocab_cache_size=config.vocab_cache_size)
                self.wide_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, 1, target=target,
                                                               slice_mode=nn.EmbeddingLookup.TABLE_ROW_SLICE,
                                                               sparse=sparse,
                                                               vocab_cache_size=config.vocab_cache_size)
            else:
                self.deep_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, self.emb_dim,
                                                               target=target, sparse=sparse,
                                                               vocab_cache_size=config.vocab_cache_size)
                self.wide_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, 1, target=target, sparse=sparse,
                                                               vocab_cache_size=config.vocab_cache_size)
            self.deep_embeddinglookup.embedding_table.set_param_ps()
            self.wide_embeddinglookup.embedding_table.set_param_ps()
        else:
            self.deep_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, self.emb_dim,
                                                           target='DEVICE', sparse=sparse,
                                                           vocab_cache_size=config.vocab_cache_size)
            self.wide_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, 1,
                                                           target='DEVICE', sparse=sparse,
                                                           vocab_cache_size=config.vocab_cache_size)

    def construct(self, id_hldr, wt_hldr):
        """
        Args:
            id_hldr: batch ids;
            wt_hldr: batch weights;
        """
        # Wide layer
        wide_id_weight = self.wide_embeddinglookup(id_hldr)
        # Deep layer
        deep_id_embs = self.deep_embeddinglookup(id_hldr)
        mask = self.reshape(wt_hldr, (self.batch_size, self.field_size, 1))
        # Wide layer
        wide_in = self.wide_mul(wide_id_weight, mask)
        wide_out = self.reshape(self.reduce_sum(wide_in, 1) + self.wide_b, (-1, 1))
        # Deep layer
        deep_in = self.deep_mul(deep_id_embs, mask)
        deep_in = self.deep_reshape(deep_in, (-1, self.field_size * self.emb_dim))
        deep_in = self.dense_layer_1(deep_in)
        deep_in = self.dense_layer_2(deep_in)
        deep_in = self.dense_layer_3(deep_in)
        deep_out = self.dense_layer_4(deep_in)
        return wide_out, deep_out


class CutLayer(nn.Cell):
    """
    Cut layer of the leader net.
    Args:
        config (class): default config info.
    """

    def __init__(self, config, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)
        self.dense_layer = DenseLayer(4, 2,
                                      config.weight_bias_init, config.deep_layer_act,
                                      use_activation=False, convert_dtype=True, drop_out=False)
        self.concat = ops.Concat(axis=1)

    def construct(self, wide_out0, deep_out0, wide_embedding, deep_embedding):
        in_ts = self.concat((wide_out0, deep_out0, wide_embedding, deep_embedding))
        return self.dense_layer(in_ts)


class TeeLayer(nn.Cell):
    """
    TEE layer of the leader net.
    Args:
        config (class): default config info.
    """
    def __init__(self, config):
        super(TeeLayer, self).__init__()
        init_tee_cut_layer(config.batch_size, 2, 2, 1, 3.5e-4, 1024.0)
        self.concat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()

    def construct(self, wide_out0, deep_out0, wide_embedding, deep_embedding):
        """Convert and encrypt the intermediate data"""
        local_emb = self.concat((wide_out0, deep_out0))
        remote_emb = self.concat((wide_embedding, deep_embedding))
        aa = remote_emb.flatten().asnumpy().tolist()
        bb = local_emb.flatten().asnumpy().tolist()
        enc_aa, enc_aa_len = encrypt_client_data(aa, len(aa))
        enc_bb, enc_bb_len = encrypt_client_data(bb, len(bb))
        tee_output = secure_forward_tee_cut_layer(remote_emb.shape[0], remote_emb.shape[1],
                                                  local_emb.shape[1], enc_aa, enc_aa_len, enc_bb, enc_bb_len, 2)
        tee_output = self.reshape(Tensor(tee_output), (remote_emb.shape[0], 2))
        return tee_output


class HeadLayer(nn.Cell):
    """
    Head layer of the leader net.
    Args:
        config (class): default config info.
    """

    def __init__(self, config, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)
        self.dense_layer = DenseLayer(2, 1,
                                      config.weight_bias_init, config.deep_layer_act,
                                      use_activation=False, convert_dtype=True, drop_out=False)

    def construct(self, layer_input):
        return self.dense_layer(layer_input)


class LeaderNet(nn.Cell):
    """
    Net of the leader party.
    Args:
        config (class): default config info.
    """

    def __init__(self, config):
        super().__init__()
        self.head_layer = HeadLayer(config)
        if hasattr(config, 'enable_TEE') and config.enable_TEE:
            self.cut_layer = TeeLayer(config)
        else:
            self.cut_layer = CutLayer(config)
        self.bottom_net = WideDeepModel(config, config.leader_field_size)

    def construct(self, id_hldr, wt_hldr, wide_embedding, deep_embedding):
        wide_out0, deep_out0 = self.bottom_net(id_hldr, wt_hldr)
        in_ts = self.cut_layer(wide_out0, deep_out0, wide_embedding, deep_embedding)
        return self.head_layer(in_ts)


class HeadLossNet(nn.Cell):
    """
    Loss net of the leader party's head net.
    Args:
        net (class): HeadLayer, which is the head net of leader party.
    """

    def __init__(self, net: HeadLayer, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)
        self.net = net
        self.loss = ops.SigmoidCrossEntropyWithLogits()
        self.reduce_mean_false = ops.ReduceMean(keep_dims=False)

    def construct(self, layer_input, label):
        out = self.net(layer_input)
        log_loss = self.loss(out, label)
        wide_loss = self.reduce_mean_false(log_loss)
        return wide_loss


class L2LossNet(nn.Cell):
    """
    The l2 regularization loss net of the leader party's head net.
    Args:
        net (class): WideDeepModel, which is the bottom net of leader party.
        config (class): default config info.
    """

    def __init__(self, net: WideDeepModel, config, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)
        self.net = net
        self.l2_coef = config.l2_coef
        self.square = ops.Square()
        self.reduce_sum_false = ops.ReduceSum(keep_dims=False)

    def construct(self):
        l2_regu = self.reduce_sum_false(
            self.square(self.net.deep_embeddinglookup.embedding_table)) / 2 * self.l2_coef
        return l2_regu


class LeaderLossNet(nn.Cell):
    """
    Train net of the leader party.
    Args:
        net (class): LeaderNet, which is the net of leader party.
        config (class): default config info.
    """

    def __init__(self, net: LeaderNet, config):
        super(LeaderLossNet, self).__init__(auto_prefix=False)
        self.net = net
        self.l2_coef = config.l2_coef
        self.loss = ops.SigmoidCrossEntropyWithLogits()
        self.square = ops.Square()
        self.reduce_mean_false = ops.ReduceMean(keep_dims=False)
        self.reduce_sum_false = ops.ReduceSum(keep_dims=False)

    def construct(self, id_hldr, wt_hldr, wide_embedding, deep_embedding, label):
        out = self.net(id_hldr, wt_hldr, wide_embedding, deep_embedding)
        log_loss = self.loss(out, label)
        wide_loss = self.reduce_mean_false(log_loss)
        l2_regu = self.reduce_sum_false(
            self.square(self.net.bottom_net.deep_embeddinglookup.embedding_table)) / 2 * self.l2_coef
        deep_loss = self.reduce_mean_false(log_loss) + l2_regu
        return out, wide_loss, deep_loss


class LeaderGradNet(nn.Cell):
    """
    Grad Network of the leader party.
    Args:
        net (class): LeaderNet, which is the net of leader party.
        config (class): default config info.
    """

    def __init__(self, net: LeaderNet, config, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)
        self.net = net
        self.sens = 1024.0
        self.enable_tee = False
        if hasattr(config, 'enable_TEE') and config.enable_TEE:
            self.enable_tee = True

        self.grad_op_param_sens = ops.GradOperation(get_by_list=True, sens_param=True)
        self.grad_op_input_sens = ops.GradOperation(get_all=True, sens_param=True)

        self.params_head = ParameterTuple(net.head_layer.trainable_params())
        if not self.enable_tee:
            self.params_cutlayer = ParameterTuple(net.cut_layer.trainable_params())
        self.params_bottom_deep = vfl_utils.get_params_by_name(self.net.bottom_net, ['deep', 'dense'])
        self.params_bottom_wide = vfl_utils.get_params_by_name(self.net.bottom_net, ['wide'])

        self.loss_net = HeadLossNet(net.head_layer)
        self.loss_net_l2 = L2LossNet(net.bottom_net, config)

        self.optimizer_head = Adam(self.params_head, learning_rate=3.5e-4, eps=1e-8, loss_scale=self.sens)
        if not self.enable_tee:
            self.optimizer_cutlayer = SGD(self.params_cutlayer, learning_rate=3.5e-4, momentum=0.8,
                                          loss_scale=self.sens)
        self.optimizer_bottom_deep = Adam(self.params_bottom_deep, learning_rate=3.5e-4,
                                          eps=1e-8, loss_scale=self.sens)
        self.optimizer_bottom_wide = FTRL(self.params_bottom_wide, learning_rate=5e-2, l1=1e-8, l2=1e-8,
                                          initial_accum=1.0, loss_scale=self.sens)

        self.print = ops.Print()
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(axis=1)

    def construct(self, local_data_batch, remote_data_batch):
        """
        The back propagation of the leader net.
        """
        # data processing
        id_hldr = local_data_batch['id_hldr']
        wt_hldr = local_data_batch['wt_hldr']
        label = local_data_batch['label']
        wide_embedding = remote_data_batch['wide_embedding']
        deep_embedding = remote_data_batch['deep_embedding']

        # forward
        wide_out0, deep_out0 = self.net.bottom_net(id_hldr, wt_hldr)
        local_emb = self.concat((wide_out0, deep_out0))
        remote_emb = self.concat((wide_embedding, deep_embedding))
        head_input = self.net.cut_layer(wide_out0, deep_out0, wide_embedding, deep_embedding)
        loss = self.loss_net(head_input, label)

        # update of head net
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), 1024.0)
        grad_head_input, _ = self.grad_op_input_sens(self.loss_net)(head_input, label, sens)
        grad_head_param = self.grad_op_param_sens(self.loss_net, self.params_head)(head_input, label, sens)
        self.optimizer_head(grad_head_param)

        # update of cut layer
        if self.enable_tee:
            tmp = grad_head_input.flatten().asnumpy().tolist()
            grad_input = backward_tee_cut_layer(remote_emb.shape[0], remote_emb.shape[1], local_emb.shape[1], 1, tmp)
            grad_inputa = self.reshape(Tensor(grad_input[0]), remote_emb.shape)
            grad_inputb = self.reshape(Tensor(grad_input[1]), local_emb.shape)
            grad_cutlayer_input = (grad_inputb[:, :1], grad_inputb[:, 1:2], grad_inputa[:, :1], grad_inputa[:, 1:2])
        else:
            grad_cutlayer_input = self.grad_op_input_sens(self.net.cut_layer)(wide_out0,
                                                                              deep_out0,
                                                                              wide_embedding,
                                                                              deep_embedding,
                                                                              grad_head_input)
            grad_cutlayer_param = self.grad_op_param_sens(self.net.cut_layer, self.params_cutlayer)(wide_out0,
                                                                                                    deep_out0,
                                                                                                    wide_embedding,
                                                                                                    deep_embedding,
                                                                                                    grad_head_input)
            self.optimizer_cutlayer(grad_cutlayer_param)

        # update of bottom net
        grad_bottom_wide = self.grad_op_param_sens(self.net.bottom_net,
                                                   self.params_bottom_wide)(id_hldr, wt_hldr,
                                                                            grad_cutlayer_input[0:2])
        self.optimizer_bottom_wide(grad_bottom_wide)
        grad_bottom_deep = self.grad_op_param_sens(self.net.bottom_net,
                                                   self.params_bottom_deep)(id_hldr, wt_hldr,
                                                                            grad_cutlayer_input[0:2])
        grad_bottom_l2 = self.grad_op_param_sens(self.loss_net_l2, self.params_bottom_deep)(sens)
        zipped = zip(grad_bottom_deep, grad_bottom_l2)
        grad_bottom_deep = tuple(map(sum, zipped))
        self.optimizer_bottom_deep(grad_bottom_deep)

        # output the gradients for follower party
        scales = {}
        scales['wide_loss'] = OrderedDict(zip(['wide_embedding', 'deep_embedding'], grad_cutlayer_input[2:4]))
        scales['deep_loss'] = scales['wide_loss']
        return scales


class LeaderEvalNet(nn.Cell):
    """
    Eval net of the leader party, which warps LeaderNet.
    Args:
        net (class): LeaderNet, which is the net of leader party.
    """

    def __init__(self, net: LeaderNet):
        super(LeaderEvalNet, self).__init__(auto_prefix=False)
        self.net = net
        self.sigmoid = ops.Sigmoid()
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        full_batch = context.get_auto_parallel_context("full_batch")
        is_auto_parallel = parallel_mode in (
            ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        if is_auto_parallel and full_batch:
            self.sigmoid.shard(((1, 1),))

    def construct(self, id_hldr, wt_hldr, wide_embedding, deep_embedding):
        logits = self.net(id_hldr, wt_hldr, wide_embedding, deep_embedding)
        pred_probs = self.sigmoid(logits)
        return logits, pred_probs


class FollowerNet(WideDeepModel):
    """
    Net of the follower party.
    Args:
        config (class): default config info.
    """

    def __init__(self, config):
        super(FollowerNet, self).__init__(config, config.follower_field_size)


class FollowerLossNet(nn.Cell):
    """
    Train net of the follower party.
    Args:
        net (class): WideDeepModel, which is the net of follower party.
        config (class): default config info.
    """

    def __init__(self, net: WideDeepModel, config):
        super(FollowerLossNet, self).__init__(auto_prefix=False)
        self.net = net
        self.l2_coef = config.l2_coef
        self.square = ops.Square()
        self.reduce_mean_false = ops.ReduceMean(keep_dims=False)

    def construct(self, id_hldr0, wt_hldr0):
        wide_embedding, deep_embedding = self.net(id_hldr0, wt_hldr0)
        l2_regu = self.reduce_mean_false(
            self.square(self.net.deep_embeddinglookup.embedding_table)) / 2 * self.l2_coef
        return wide_embedding, deep_embedding, l2_regu
