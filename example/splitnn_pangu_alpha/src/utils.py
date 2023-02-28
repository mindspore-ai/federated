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
"""
network config setting, gradient clip function and dynamic learning rate function
"""
import argparse
import ast
import os
import time
import numpy as np

from mindspore import context
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.communication.management as D
from mindspore.nn.transformer import TransformerOpParallelConfig, CrossEntropyLoss, TransformerRecomputeConfig
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.communication.management import get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs

from src.dataset import create_dataset
from src.pangu_alpha_config import PanguAlphaConfig

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


def _get_model_parallel_group(mp):
    """

    Calculate the communication group of model parallel dim in one pipeline stage

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    stage_id = rank // per_stage_device_nums
    local_stage_rank_id = rank % per_stage_device_nums
    index = local_stage_rank_id // mp
    group = range(0, mp)
    rank_str_list = [str(x + index * mp + stage_id * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    rank_list = [x + index * mp + stage_id * per_stage_device_nums for x in group]
    return rank_list, rank_list_str


def _get_pipeline_group():
    """

    Calculate the communication group between all pipeline stages

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    local_stage_rank_id = rank % per_stage_device_nums
    group = range(0, stage_nums)
    rank_list = [local_stage_rank_id + x * per_stage_device_nums for x in group]
    rank_str_list = [str(local_stage_rank_id + x * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    return rank_list, rank_list_str


def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    return group_params


def set_embedding_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'order_params': params
    }]
    return group_params


def add_checkpoint_callback_policy(args_param, callback, rank_id):
    r"""
    Add checkpoint policy to callback.
    """
    if rank_id == 0:
        ckpt_append_info = [
            {"epoch_num": args_param.has_trained_epoches, "step_num": args_param.has_trained_steps, "global_step": 0}]
        ckpt_config = CheckpointConfig(save_checkpoint_steps=1000000,
                                       keep_checkpoint_max=1,
                                       integrated_save=True,
                                       append_info=ckpt_append_info)
        # save checkpoint into rank directory
        ckpoint_cb = ModelCheckpoint(prefix=args_param.ckpt_name_prefix + str(rank_id),
                                     directory="/tmp/ckpt/",
                                     config=ckpt_config)
        callback.append(ckpoint_cb)


def set_parallel_context(args_opt):
    r"""Set parallel context"""
    D.init()
    device_num = D.get_group_size()
    rank = D.get_rank()
    print("rank_id is {}, device_num is {}".format(rank, device_num))
    context.reset_auto_parallel_context()
    if args_opt.mode == "350M":
        if device_num == 1:
            context.set_auto_parallel_context(
                parallel_mode=context.ParallelMode.STAND_ALONE,
                gradients_mean=False,
                full_batch=bool(args_opt.full_batch),
                enable_parallel_optimizer=False,
                enable_alltoall=False)
        else:
            context.set_auto_parallel_context(
                parallel_mode=context.ParallelMode.SEMI_AUTO_PARALLEL,
                gradients_mean=False,
                full_batch=bool(args_opt.full_batch),
                enable_parallel_optimizer=False,
                enable_alltoall=False)
    else:
        context.set_auto_parallel_context(
            parallel_mode=args_opt.parallel_mode,
            gradients_mean=False,
            full_batch=bool(args_opt.full_batch),
            strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path,
            enable_parallel_optimizer=bool(args_opt.optimizer_shard),
            strategy_ckpt_save_file='strategy.ckpt',
            enable_alltoall=bool(args_opt.enable_alltoall))
    set_algo_parameters(elementwise_op_strategy_follow=True)
    _set_multi_subgraphs()
    return rank, device_num


def load_train_net(args_opt):
    r"""The main training process."""
    rank = 0
    device_num = 1
    if args_opt.distribute == "true":
        rank, device_num = set_parallel_context(args_opt)
    context.set_context(save_graphs=False, save_graphs_path="./graphs_of_device_id_" + str(rank))
    if args_opt.parallel_mode == "data_parallel":
        # in avoid of the loop call depth
        context.set_context(max_call_depth=10000)

    # env variable prepare
    group_info_file = os.getenv("GROUP_INFO_FILE")
    if group_info_file:
        with open(os.path.expanduser("job/code/group_info_env"), "a") as outfile:
            outfile.write(f"export GROUP_INFO_FILE_REFLECT={group_info_file}\n")

    # Set model property
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    batch_size = args_opt.per_batch_size * data_parallel_num if context.get_auto_parallel_context(
        "parallel_mode") != context.ParallelMode.DATA_PARALLEL else args_opt.per_batch_size
    micro_batch_interleaved = args_opt.micro_batch_interleaved
    recompute_config = TransformerRecomputeConfig(recompute=True,
                                                  recompute_slice_activation=bool(args_opt.recompute_slice_activation))
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num,
                                                  expert_parallel=args_opt.expert_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  optimizer_shard=bool(args_opt.optimizer_shard),
                                                  vocab_emb_dp=bool(args_opt.word_emb_dp), recompute=recompute_config,
                                                  gradient_aggregation_group=args_opt.gradient_aggregation_group)
    config = PanguAlphaConfig(batch_size=batch_size // micro_batch_interleaved,
                              num_heads=args_opt.num_heads,
                              hidden_size=args_opt.embedding_size,
                              seq_length=args_opt.seq_length,
                              vocab_size=args_opt.vocab_size,
                              num_layers=args_opt.num_layers,
                              eod_token=args_opt.eod_id,
                              ffn_hidden_size=args_opt.embedding_size * 4,
                              eod_reset=bool(args_opt.eod_reset),
                              load_ckpt_path=args_opt.load_ckpt_path,
                              expert_num=args_opt.expert_num,
                              param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16,
                              enable_offload=bool(args_opt.opt_offload),
                              use_moe=bool(args_opt.use_moe),
                              per_token_num_experts_chosen=args_opt.per_token_num_experts_chosen,
                              hidden_act='fast_gelu' if args_opt.device_target != "GPU" else 'gelu',
                              parallel_config=parallel_config)
    print("===config is: ", config, flush=True)
    loss = CrossEntropyLoss(config.parallel_config.dp_mp_config)
    return loss, config


def construct_local_dataset(args_opt, rank_id, device_num, is_training=True):
    """create dataset object according to config info."""
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    batch_size = args_opt.per_batch_size * data_parallel_num
    micro_batch_interleaved = args_opt.micro_batch_interleaved
    if is_training:
        ds = create_dataset(batch_size * micro_batch_interleaved,
                            data_path=args_opt.data_url,
                            data_start_index=0,  # max(0, epoch_idx-1),
                            eod_reset=args_opt.eod_reset,
                            full_batch=bool(args_opt.full_batch),
                            eod_id=args_opt.eod_id,
                            device_num=device_num,
                            rank=rank_id,
                            column_name=args_opt.data_column_name,
                            epoch=1)
        print("===train dataset size: ", int(ds.get_dataset_size()), flush=True)
    else:
        ds = create_dataset(batch_size * micro_batch_interleaved, data_path=args_opt.eval_data_url, data_start_index=0,
                            eod_reset=args_opt.eod_reset, full_batch=bool(args_opt.full_batch), eod_id=args_opt.eod_id,
                            device_num=device_num,
                            rank=rank_id,
                            column_name=args_opt.data_column_name,
                            epoch=1)
        print("===test dataset size: ", int(ds.get_dataset_size()), flush=True)
    return ds


class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for PanguAlpha network.
    """

    def __init__(self,
                 learning_rate,
                 end_learning_rate,
                 warmup_steps,
                 decay_steps,
                 power=1.0,
                 use_cosine=True):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate,
                                          decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, learning_rate,
                                             decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step),
                                  mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


def add_inference_params(opt):
    """Add inference params"""
    opt.add_argument("--frequency_penalty",
                     type=float,
                     default=1.5,
                     help="coefficient for frequency_penalty")
    opt.add_argument("--presence_penalty",
                     type=float,
                     default=0.3,
                     help="coefficient for presence_penalty")
    opt.add_argument("--max_generate_length",
                     type=int,
                     default=500,
                     help="the maximum number of generated token")
    opt.add_argument("--top_k_num",
                     type=int,
                     default=3,
                     help="the number for top_k sampling")
    opt.add_argument("--top_p",
                     type=float,
                     default=1.0,
                     help="top_p sampling threshold, enabled if less than 1.0")
    opt.add_argument("--end_token",
                     type=int,
                     default=9,
                     help="the token id for <end of document>")
    opt.add_argument("--use_pynative_op",
                     type=int,
                     default=0,
                     help="Whether use pynative op for postproecess")
    opt.add_argument("--use_past",
                     type=str,
                     default="true",
                     choices=["true", "false"],
                     help="Whether enable state reuse")


def add_training_params(opt):
    """Add training params"""
    opt.add_argument("--http_server_address",
                     type=str, default="127.0.0.1:5555",
                     help="the address of local training server")
    opt.add_argument("--remote_server_address",
                     type=str, default="127.0.0.1:7777",
                     help="the address of remote training server")
    opt.add_argument("--seq_length",
                     type=int, default=1024,
                     help="sequence length, default is 1024.")
    opt.add_argument("--vocab_size",
                     type=int, default=40000,
                     help="vocabulary size, default is 40000.")
    opt.add_argument("--embedding_size",
                     type=int, default=16384,
                     help="embedding table size, default is 16384.")
    opt.add_argument("--num_layers",
                     type=int, default=64,
                     help="total layers, default is 64.")
    opt.add_argument("--num_heads", type=int, default=128, help="head size, default is 128.")
    opt.add_argument("--stage_num", type=int, default=1, help="Pipeline stage num, default is 1.")
    opt.add_argument("--micro_size",
                     type=int, default=1,
                     help="Pipeline micro_size, default is 1.")
    opt.add_argument("--eod_reset",
                     type=int, default=1,
                     help="Enable eod mask, default is 1.")
    opt.add_argument("--warmup_step",
                     type=int, default=2000,
                     help="Warmup step, default is 2000.")
    opt.add_argument("--decay_steps",
                     type=int, default=200000,
                     help="Decay step, default is 200000.")
    opt.add_argument("--optimizer",
                     type=str, default="adam",
                     choices=["adam", "lamb"],
                     help="select which optimizer to be used, default adam")
    opt.add_argument("--opt_offload",
                     type=int, default=0,
                     help="Enable optimizer status offload to host CPU, default is 0")
    opt.add_argument("--use_moe",
                     type=int, default=0,
                     help="Use moe, default is 0")
    opt.add_argument("--expert_num",
                     type=int, default=1,
                     help="Expert number, only effective when applying moe, Default is 1")
    opt.add_argument("--per_token_num_experts_chosen",
                     type=int, default=1,
                     help="Expert nums chosen by each token, only effective when applying moe, default is 1")
    opt.add_argument("--eod_id",
                     type=int, default=6,
                     help="The id of end of document")
    opt.add_argument("--padding_id",
                     type=int, default=6,
                     help="The padding id of dataset")
    opt.add_argument("--epoch_size",
                     type=int, default=1,
                     help="The training epoch")
    opt.add_argument("--sink_size",
                     type=int, default=2,
                     help="The sink size of the training. default is 2")
    opt.add_argument("--full_batch",
                     default=1, type=int,
                     help="Import the full size of a batch for each card, default is 1")
    opt.add_argument("--optimizer_shard",
                     type=int, default=0,
                     help="Enable optimizer parallel, default is 0")
    opt.add_argument("--per_batch_size",
                     type=int, default=2,
                     help="The batch size for each data parallel way. default 0")
    opt.add_argument("--start_lr",
                     type=float, default=5e-5,
                     help="The start learning rate. default 5e-5")
    opt.add_argument("--end_lr",
                     type=float, default=1e-6,
                     help="The end learning rate. default 1e-6")
    opt.add_argument("--op_level_model_parallel_num",
                     type=int, default=8,
                     help="The model parallel way. default 8")
    opt.add_argument("--expert_parallel_num",
                     type=int, default=1,
                     help="The expert parallel way, only effective when applying moe. Default 1")
    opt.add_argument("--word_emb_dp",
                     type=int, default=1,
                     choices=[0, 1],
                     help="Whether do data parallel in word embedding. default 1")
    opt.add_argument("--gradient_aggregation_group",
                     type=int, default=4,
                     help="The gradient communication fusion group. default 4")
    opt.add_argument("--data_column_name",
                     type=str, default="input_ids",
                     help="Column name of datasets")
    opt.add_argument("--micro_batch_interleaved",
                     type=int, default=1,
                     help="Parallel split num of batch size. default 2")
    opt.add_argument("--recompute_slice_activation",
                     type=int, default=0,
                     help="Enable slice the recompute activation state. default 0")


def add_context_args_mode(opt):
    """Add context args params"""
    opt.add_argument("--parallel_mode",
                     type=str,
                     default="stand_alone",
                     choices=['data_parallel', "semi_auto_parallel", "auto_parallel", "stand_alone"],
                     help="The parallel context mode")


def add_retrain_params(opt):
    """
    Add parameters about retrain.
    """
    opt.add_argument("--resume",
                     type=ast.literal_eval,
                     default=False,
                     help="Whether to resume the pretrained model.")
    opt.add_argument("--pre_trained_embedding",
                     type=str,
                     default=None,
                     help="Pretrained embedding checkpoint path.")
    opt.add_argument("--pre_trained_backbone",
                     type=str,
                     default=None,
                     help="Pretrained backbone checkpoint path.")
    opt.add_argument("--pre_trained_head",
                     type=str,
                     default=None,
                     help="Pretrained head checkpoint path.")
    opt.add_argument("--pre_trained",
                     type=str,
                     default=None,
                     help="Pretrained checkpoint path.")
    opt.add_argument("--save_checkpoint_path",
                     type=str,
                     default=None,
                     help="Save checkpoint path.")
    opt.add_argument("--keep_checkpoint_max",
                     type=int,
                     default=1,
                     help="Max checkpoint save number.")
    opt.add_argument("--save_checkpoint_steps",
                     type=int,
                     default=2000,
                     help="Save checkpoint step number.")
    opt.add_argument("--save_checkpoint",
                     type=ast.literal_eval,
                     default=False,
                     help="Whether save checkpoint in local disk.")
    opt.add_argument("--ckpt_name_prefix",
                     type=str,
                     default="pangu",
                     help="Saving checkpoint name prefix.")
    opt.add_argument("--has_trained_epoches",
                     type=int,
                     default=0,
                     help="Epoches has been trained before.")
    opt.add_argument("--has_trained_steps",
                     type=int,
                     default=0,
                     help="Steps has been trained before.")


def add_privacy_params(opt):
    opt.add_argument("--embedding_dp",
                     type=bool,
                     default=False,
                     help="Whether apply Embedding DP.")


def get_args(inference=False):
    """train function for PanguAlpha"""
    parser = argparse.ArgumentParser(description="PanguAlpha training")
    parser.add_argument('--device_id',
                        type=int,
                        default=0,
                        help="Device id, default is 0.")
    parser.add_argument("--device_num",
                        type=int,
                        default=1,
                        help="Use device nums, default is 1.")
    parser.add_argument("--distribute",
                        type=str,
                        default="false",
                        choices=["true", "false"],
                        help="Run distribute, default is true.")
    parser.add_argument("--load_ckpt_name",
                        type=str,
                        default='PANGUALPHA3.ckpt',
                        help="checkpint file name.")
    parser.add_argument("--load_ckpt_path",
                        type=str,
                        default=None,
                        help="predict file path.")
    parser.add_argument('--data_url',
                        required=False,
                        default='./wiki/train/',
                        help='Location of train data.')
    parser.add_argument('--eval_data_url',
                        required=False,
                        default='./wiki/test/',
                        help='Location of eval data.')
    parser.add_argument('--train_url',
                        required=False,
                        default=None,
                        help='Location of training outputs.')
    parser.add_argument("--run_type",
                        type=str,
                        default="train",
                        choices=["train", "predict"],
                        help="The run type")
    parser.add_argument("--mode",
                        type=str,
                        default="350M",
                        choices=["200B", "13B", "2.6B", "1.3B", "350M", "self_define"],
                        help="The scale of the model parameters")
    parser.add_argument("--device_target",
                        type=str,
                        default="GPU",
                        choices=["Ascend", "GPU"],
                        help="The running device")
    parser.add_argument("--strategy_load_ckpt_path",
                        type=str,
                        default="",
                        help="The training prallel strategy for the model.")
    parser.add_argument("--tokenizer_path",
                        type=str,
                        default="./tokenizer_path",
                        help="The path where stores vocab and vocab model file")
    parser.add_argument("--param_init_type",
                        type=str,
                        default="fp32",
                        help="The initialization type for parameters. Default fp16.")
    parser.add_argument("--offline",
                        type=int,
                        default=1,
                        help="Running on cloud of not. Default 1.")
    parser.add_argument("--export",
                        type=int,
                        default=0,
                        help="Whether export mindir for serving.")
    parser.add_argument("--incremental_training",
                        type=int,
                        default=0,
                        help="Enable incremental training. Default 0.")
    parser.add_argument("--train_and_eval_mode",
                        type=int,
                        default=0,
                        help="Enable evaling while training. Default 0.")
    parser.add_argument("--eval_steps",
                        type=int,
                        default=10,
                        help="The eval step in train and eval mode. Default 10.")
    parser.add_argument("--enable_alltoall",
                        type=int,
                        default=0,
                        help="Enable alltoall communication, only effective when applying moe. Default 0")
    parser.add_argument("--hccl_connect_time",
                        type=int,
                        default=6000,
                        help="Set the hccl build time out, only effective on Ascend. Default 6000")
    parser.add_argument("--ng_port", type=str, default='33222', help="grpc client  port")
    add_context_args_mode(parser)
    add_training_params(parser)
    add_retrain_params(parser)
    add_privacy_params(parser)
    if inference:
        add_inference_params(parser)
    args_opt = parser.parse_args()

    return args_opt


def download_data(src_data_url, tgt_data_path, rank):
    """
        Download the dataset from the obs.
        src_data_url (Str): should be the dataset path in the obs
        tgt_data_path (Str): the local dataset path
        rank (Int): the current rank id

    """
    cache_url = tgt_data_path
    exec_path = '/tmp'
    if rank % 8 == 0:
        import moxing as mox
        print("Modify the time out from 300 to 30000")
        print("begin download dataset", flush=True)

        if not os.path.exists(cache_url):
            os.makedirs(cache_url, exist_ok=True)
        mox.file.copy_parallel(src_url=src_data_url,
                               dst_url=cache_url)
        print("Dataset download succeed!", flush=True)

        f = open("%s/install.txt" % exec_path, 'w')
        f.close()
    # stop
    while not os.path.exists("%s/install.txt" % exec_path):
        time.sleep(1)
