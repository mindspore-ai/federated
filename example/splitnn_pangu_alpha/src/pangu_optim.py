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
"""
Customized Adam optimizer for pangu_alpha training
"""
from mindspore import context, ops, Tensor
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore.common import Parameter, ParameterTuple
from mindspore.common.initializer import initializer
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell

from mindspore_federated import vfl_utils

from src.adam import AdamWeightDecayOp


def set_optimizer(optimizer, opt_offload, group_params, learning_rate, config):
    r"""Set optimizer"""
    if optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=learning_rate)
    elif opt_offload:
        optimizer = AdamWeightDecayOp(group_params, learning_rate=learning_rate, eps=1e-8, beta1=0.9, beta2=0.95,
                                      param_init_type=config.param_init_type)
    else:
        optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=learning_rate, eps=1e-8, beta1=0.9, beta2=0.95)
    return optimizer


class FP32StateAdamWeightDecay(nn.AdamWeightDecay):
    r"""
        This class is almost same with the mindspore's AdamWeightDecay implements, the
        only difference is the optimizer's state will be always initialized with float32,
        where the original AdamWeightDecay will initialize the optimizer's state with float16,
        if the parameters are initialized with fp16.
        This setting will avoid overflow in training PanGu-Alpha model using fp16.
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(FP32StateAdamWeightDecay, self).__init__(params, learning_rate=learning_rate,
                                                       beta1=beta1,
                                                       beta2=beta2,
                                                       eps=eps,
                                                       weight_decay=weight_decay)

        self.moments1 = self.clone_state(self.parameters, prefix='adam_m', init='zeros')
        self.moments2 = self.clone_state(self.parameters, prefix='adam_v', init='zeros')

    def clone_state(self, parameter_tuple, prefix, init):
        r"""
            parameter_tuple: ParameterTuple. The parameters of the network
            prefix: str. The prefix name of the parameters
            init: str. The initialization method
        """
        new = []
        for old_param in parameter_tuple:
            new_state = Parameter(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.param_info = old_param.param_info.clone()
            new_state.is_init = False
            new_state.set_data(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ParameterTuple(new)


class PanguAlphaAdam(TrainOneStepWithLossScaleCell):
    """
    Customized Adam optimizer for training of pangu_alpha in the splitnn demo system.
    """
    def __init__(self, net, optim_inst, scale_update_cell, config, yaml_data) -> None:
        super(PanguAlphaAdam, self).__init__(net, optim_inst, scale_update_cell)
        self.net = net
        self.optim_yaml = yaml_data.opts[0]
        self.net_yaml = yaml_data.train_net
        self.weights = optim_inst.parameters
        self.optimizer = optim_inst

        self.grad_list = []
        self.loss_net_list = []
        self.sens_list = []
        for grad_yaml in self.optim_yaml['grads']:
            grad_inst = C.GradOperation(get_by_list=True, sens_param=True)
            output_name = grad_yaml['output']['name']
            if len(self.net_yaml['outputs']) > 1:
                for idx, output in enumerate(self.net_yaml['outputs']):
                    if output['name'] == output_name:
                        output_index = idx
                        break
                loss_net = vfl_utils.IthOutputCellInTuple(self.net, output_index)
            else:
                loss_net = self.net
            loss_net.set_grad()
            self.loss_net_list.append(loss_net)
            self.grad_list.append(grad_inst(loss_net, self.weights))
            self.sens_list.append('sens' in grad_yaml)
        self._loss_dtype = None
        self._loss_shape = None
        self._dtype_op = ops.DType()
        self._shape_op = ops.Shape()
        self._fill_op = ops.Fill()
        self.degree = 1
        self.enable_global_norm = True
        self.enable_offload = config.enable_offload
        self.clip_value = Tensor([1.0], dtype=mstype.float32)
        self.cast = P.Cast()
        self._init_clip_global_norm(self.weights, config)

    def _init_clip_global_norm(self, params, config, clip_norm=1.0):
        """
        init the attributes on ClipGlobalNorm op
        """
        self.apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()
        self.great_equal_op = P.GreaterEqual()
        self._init_global_norm(params, config)

    def _init_global_norm(self, params, config):
        """
        init the attributes on GlobalNorm op
        """
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        self.is_pipeline = context.get_auto_parallel_context("pipeline_stages") > 1
        self.is_data_parallel = context.get_auto_parallel_context("parallel_mode") == context.ParallelMode.DATA_PARALLEL
        self.config = config
        self.group_size = 1
        self.merge_op = P.identity()
        self.group_size = 1
        self.allreduce_group_size = ()
        self.allreduce_group_size = self._get_scale_for_gradient_norm(params)

    def __call__(self, *inputs, sens=None):
        grads_yaml = self.optim_yaml['grads']
        grad_sum = tuple()
        for idx, grad_inst in enumerate(self.grad_list):
            scaling_sens = self.scale_sense
            if not self.gpu_target:
                loss_value = self.loss_net_list[idx](*inputs)
                status, scaling_sens = self.start_overflow_check(loss_value, scaling_sens)
            else:
                status = False
            if self.sens_list[idx]:
                if isinstance(self.optim_yaml['grads'][idx]['sens'], (float, int)):
                    if not self._loss_dtype and not self._loss_shape:
                        loss_value = self.loss_net_list[idx](*inputs)
                        self._loss_dtype = self._dtype_op(loss_value)
                        self._loss_shape = self._shape_op(loss_value)
                    sens_value = self._fill_op(self._loss_dtype, self._loss_shape, grads_yaml[idx]['sens'])
                elif self.optim_yaml['grads'][idx]['sens'] is not None and isinstance(grads_yaml[idx]['sens'], str):
                    sens_value = sens[grads_yaml[idx]['sens']]
                    sens_value = sens_value[grads_yaml[idx]['output']['name']]
            else:
                if not self._loss_dtype and not self._loss_shape:
                    loss_value = self.loss_net_list[idx](*inputs)
                    self._loss_dtype = self._dtype_op(loss_value)
                    self._loss_shape = self._shape_op(loss_value)
                sens_value = self._fill_op(self._loss_dtype, self._loss_shape, 1.0)
            grads = grad_inst(*inputs, sens_value)
            grads = self.grad_reducer(grads)
            grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads)
            # sum res of multiple GradOperations
            if not grad_sum:
                grad_sum = grads
            else:
                zipped = zip(grads, grad_sum)
                grad_sum = tuple(map(sum, zipped))

        clip_value = self.clip_value
        if self.enable_global_norm:
            grad_sum, clip_value = self.clip(grad_sum)
        # Check whether overflow
        cond = self.get_overflow_status(status, grad_sum)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            if self.enable_offload:
                self.optimizer(grad_sum, clip_value)
            else:
                self.optimizer(grad_sum)

    def clip(self, grads):
        grads, global_norm_value = self.global_norm(grads)
        cond = self.great_equal_op(global_norm_value, self.clip_norm)
        global_norm = F.select(cond, global_norm_value, self.clip_norm)
        grads = self.hyper_map(F.partial(apply_global_norm, False, self.clip_norm, global_norm), grads)
        return grads, global_norm_value

    def global_norm(self, grads):
        square_sum = self.hyper_map(get_square_sum, grads, self.allreduce_group_size)
        square_reduce_sum = F.addn(square_sum)
        global_norms = F.sqrt(self.merge_op(square_reduce_sum))
        return grads, global_norms

    def _get_scale_for_gradient_norm(self, params):
        """
        get scale of gradient after global normalization considering allreduce scenarios
        """
        allreduce_group_size = ()
        for x in params:
            if "projection.bias" not in x.name and "layernorm" not in x.name and "embedding_table" not in x.name:
                allreduce_group_size = allreduce_group_size + (1.0,)
            elif "embedding_table" not in x.name:
                allreduce_group_size = allreduce_group_size + (self.group_size * 1.0,)
            else:
                if not self.config.parallel_config.vocab_emb_dp and "position_embedding.embedding_table" not in x.name \
                        and "top_query_embedding_table" not in x.name:
                    allreduce_group_size = allreduce_group_size + (self.config.parallel_config.data_parallel * 1.0,)
                else:
                    allreduce_group_size = allreduce_group_size + (self.group_size * 1.0,)
        return allreduce_group_size


get_square_sum = C.MultitypeFuncGraph("get_square_sum")


@get_square_sum.register("Tensor", "Number")
def _get_square_sum(grad, value):
    norm = P.ReduceSum(False)(F.square(grad), ()) / value
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@apply_global_norm.register("Bool", "Tensor", "Tensor", "Tensor")
def _apply_global_norm(enable_grad_fp16, clip_norm, global_norm, grad):
    if enable_grad_fp16:
        grad = P.Cast()(grad * clip_norm / global_norm, mstype.float16)
    else:
        grad = grad * clip_norm / global_norm
    return grad


grad_scale = C.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * P.Reciprocal()(scale)
