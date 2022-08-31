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
"""This module realize the wide and deep network for splitnn test on criteo dataset"""

from sklearn.metrics import roc_auc_score

from mindspore import nn, context, ops
from mindspore import Parameter, Tensor
import mindspore.common.dtype as mstype
from mindspore.nn import Dropout
from mindspore.nn.metrics import Metric
from mindspore.common.initializer import Uniform, initializer
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size


def init_method(method, shape, name, max_val=1.0):
    """
    parameter init method
    Inputs:
        method (str): type of init method, including uniform, one, zero, and normal.
        shape (Union[tuple, list, int]): shape of the tensor to be initialized.
        name (str): name of the tensor to be initialized.
        max_val (float): the bound of the Uniform distribution. Default: 0.07.
    """
    if method in ['uniform']:
        params = Parameter(initializer(Uniform(max_val), shape, mstype.float32), name=name)
    elif method == "one":
        params = Parameter(initializer("ones", shape, mstype.float32), name=name)
    elif method == 'zero':
        params = Parameter(initializer("zeros", shape, mstype.float32), name=name)
    elif method == "normal":
        params = Parameter(initializer("normal", shape, mstype.float32), name=name)
    return params


def init_var_dict(init_args, in_vars):
    """
    var init function
    Args:
        init_args (list): args used to initialize variables
        in_vars (list): info. of vars to be initialized, including name, shape, and init method.
    """
    var_map = {}
    _, max_val = init_args
    for item in in_vars:
        key, shape, method = item
        if key not in var_map.keys():
            if method in ['random', 'uniform']:
                var_map[key] = Parameter(initializer(Uniform(max_val), shape, mstype.float32), name=key)
            elif method == "one":
                var_map[key] = Parameter(initializer("ones", shape, mstype.float32), name=key)
            elif method == "zero":
                var_map[key] = Parameter(initializer("zeros", shape, mstype.float32), name=key)
            elif method == 'normal':
                var_map[key] = Parameter(initializer("normal", shape, mstype.float32), name=key)
    return var_map


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
        self.dense_layer_4 = DenseLayer(self.all_dim_list[3], self.all_dim_list[4],
                                        config.weight_bias_init, config.deep_layer_act,
                                        convert_dtype=True, drop_out=config.dropout_flag)
        self.dense_layer_5 = DenseLayer(self.all_dim_list[4], self.all_dim_list[5],
                                        config.weight_bias_init, config.deep_layer_act,
                                        use_activation=False, convert_dtype=True,
                                        drop_out=config.dropout_flag)
        self.wide_mul = ops.Mul()
        self.deep_mul = ops.Mul()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.reshape = ops.Reshape()
        self.deep_reshape = ops.Reshape()
        self.concat = ops.Concat(axis=1)

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
        deep_in = self.dense_layer_4(deep_in)
        deep_out = self.dense_layer_5(deep_in)
        return wide_out, deep_out


class LeaderNet(nn.Cell):
    """
    Net of the leader party.
    Args:
        config (class): default config info.
    """

    def __init__(self, config):
        super(LeaderNet, self).__init__()
        self.dense_layer = DenseLayer(4, 1,
                                      config.weight_bias_init, config.deep_layer_act,
                                      use_activation=False, convert_dtype=True, drop_out=False)
        self.concat = ops.Concat(axis=1)
        self.bottom_net = WideDeepModel(config, config.leader_field_size)

    def construct(self, id_hldr, wt_hldr, wide_embedding, deep_embedding):
        wide_out0, deep_out0 = self.bottom_net(id_hldr, wt_hldr)
        in_ts = self.concat((wide_out0, deep_out0, wide_embedding, deep_embedding))
        return self.dense_layer(in_ts)


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
        self.l2_coef = Parameter(Tensor(config.l2_coef),
                                 name="LeaderLossNet_l2_coef",
                                 requires_grad=False)
        self.config = config
        self.loss = ops.SigmoidCrossEntropyWithLogits()
        self.square = ops.Square()
        self.reduce_mean_false = ops.ReduceMean(keep_dims=False)
        self.reduce_sum_false = ops.ReduceSum(keep_dims=False)
        self.scale_op = ops.GradOperation(get_all=True)

    def construct(self, id_hldr, wt_hldr, wide_embedding, deep_embedding, label):
        out = self.net(id_hldr, wt_hldr, wide_embedding, deep_embedding)
        log_loss = self.loss(out, label)
        wide_loss = self.reduce_mean_false(log_loss)
        l2_regu = self.reduce_sum_false(
            self.square(self.net.bottom_net.deep_embeddinglookup.embedding_table)) / 2 * 8.e-5
        deep_loss = self.reduce_mean_false(log_loss) + l2_regu
        return out, wide_loss, deep_loss


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
        self.l2_coef = Parameter(Tensor(config.l2_coef),
                                 name="FollowerLossNet_l2_coef",
                                 requires_grad=False)
        self.square = ops.Square()
        self.reduce_mean_false = ops.ReduceMean(keep_dims=False)

    def construct(self, id_hldr0, wt_hldr0):
        wide_embedding, deep_embedding = self.net(id_hldr0, wt_hldr0)
        l2_regu = self.reduce_mean_false(
            self.square(self.net.deep_embeddinglookup.embedding_table)) / 2 * 8.e-5
        return wide_embedding, deep_embedding, l2_regu
