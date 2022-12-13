# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

# pylint: disable=unused-argument
"""train resnet."""
import datetime
import glob
import os
import time
import numpy as np

import mindspore as ms
import mindspore.log as logger
import mindspore.nn as nn
from mindspore.communication.management import init
from mindspore.parallel import set_algo_parameters
from mindspore.train.callback import Callback
from mindspore.train.callback import LossMonitor
from mindspore_federated import FederatedLearningManager

from src.CrossEntropySmooth import CrossEntropySmooth
from src.dataset import create_dataset1 as create_dataset
from src.lr_generator import get_lr
from src.metric import DistAccuracy, ClassifyCorrectCell
from src.model_utils.config import config
from src.resnet import conv_variance_scaling_initializer
from src.resnet import resnet50 as resnet

ms.set_seed(1)

http_server_address = config.http_server_address
device_id = config.device_id
device_num = config.device_num


class TimeMonitor(Callback):
    """
    Time monitor for calculating cost of each epoch.
    Args
        data_size (int) step size of an epoch.
    """

    def __init__(self, data_size):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / self.data_size
        fps = config.batch_size / per_step_mseconds * 1000
        print("epoch time: {0}, per step time: {1}, fps:{2}".format(epoch_mseconds, per_step_mseconds, fps), flush=True)

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        step_mseconds = (time.time() - self.step_time) * 1000
        print(f"step time {step_mseconds}", flush=True)


class LossCallBack(LossMonitor):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    """

    def __init__(self, has_trained_epoch=0):
        super(LossCallBack, self).__init__()
        self.has_trained_epoch = has_trained_epoch

    def step_end(self, run_context):
        """
        step_end
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], ms.Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            # pylint: disable=line-too-long
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num + int(self.has_trained_epoch),
                                                      cur_step_in_epoch, loss), flush=True)


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def set_parameter():
    """set_parameter"""
    target = config.device_target
    # init context
    if config.mode_name == 'GRAPH':
        ms.set_context(mode=ms.GRAPH_MODE, device_target=target, device_id=device_id)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target, device_id=device_id)

    if config.run_distribute:
        if target == "Ascend":
            ms.set_auto_parallel_context(device_num=device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            if config.net_name == "resnet50":
                if config.boost_mode not in ["O1", "O2"]:
                    ms.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)
            init()
        # GPU target
        else:
            init()
            ms.set_auto_parallel_context(device_num=device_num,
                                         parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True, all_reduce_fusion_config=config.all_reduce_fusion_config)


def load_pre_trained_checkpoint():
    """
    Load checkpoint according to pre_trained path.
    """
    param_dict = None
    if config.pre_trained:
        if os.path.isdir(config.pre_trained):
            ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path, "ckpt_0")
            ckpt_pattern = os.path.join(ckpt_save_dir, "*.ckpt")
            ckpt_files = glob.glob(ckpt_pattern)
            if not ckpt_files:
                logger.warning(f"There is no ckpt file in {ckpt_save_dir}, "
                               f"pre_trained is unsupported.")
            else:
                ckpt_files.sort(key=os.path.getmtime, reverse=True)
                time_stamp = datetime.datetime.now()
                print(f"time stamp {time_stamp.strftime('%Y.%m.%d-%H:%M:%S')}"
                      f" pre trained ckpt model {ckpt_files[0]} loading",
                      flush=True)
                param_dict = ms.load_checkpoint(ckpt_files[0])
        elif os.path.isfile(config.pre_trained):
            param_dict = ms.load_checkpoint(config.pre_trained)
        else:
            print(f"Invalid pre_trained {config.pre_trained} parameter.")
    return param_dict


def init_weight(net, param_dict):
    """init_weight"""
    if config.pre_trained:
        if param_dict:
            if param_dict.get("epoch_num") and param_dict.get("step_num"):
                config.has_trained_epoch = int(param_dict["epoch_num"].data.asnumpy())
                config.has_trained_step = int(param_dict["step_num"].data.asnumpy())
            else:
                config.has_trained_epoch = 0
                config.has_trained_step = 0

            if config.filter_weight:
                filter_list = [x.name for x in net.end_point.get_parameters()]
                filter_checkpoint_parameter_by_list(param_dict, filter_list)
            ms.load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                if config.conv_init == "XavierUniform":
                    cell.weight.set_data(ms.common.initializer.initializer(ms.common.initializer.XavierUniform(),
                                                                           cell.weight.shape,
                                                                           cell.weight.dtype))
                elif config.conv_init == "TruncatedNormal":
                    weight = conv_variance_scaling_initializer(cell.in_channels,
                                                               cell.out_channels,
                                                               cell.kernel_size[0])
                    cell.weight.set_data(weight)
            if isinstance(cell, nn.Dense):
                if config.dense_init == "TruncatedNormal":
                    cell.weight.set_data(ms.common.initializer.initializer(ms.common.initializer.TruncatedNormal(),
                                                                           cell.weight.shape,
                                                                           cell.weight.dtype))
                elif config.dense_init == "RandomNormal":
                    in_channel = cell.in_channels
                    out_channel = cell.out_channels
                    weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
                    weight = ms.Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=cell.weight.dtype)
                    cell.weight.set_data(weight)


def init_lr(step_size):
    """init lr"""
    if config.optimizer == "Thor":
        from src.lr_generator import get_thor_lr
        lr = get_thor_lr(0, config.lr_init, config.lr_decay, config.lr_end_epoch, step_size, decay_epochs=39)
    else:
        lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                    warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=step_size,
                    lr_decay_mode=config.lr_decay_mode)
    return lr


def init_loss_scale():
    if config.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    return loss


def init_group_params(net):
    """init group params"""
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params


def train_net():
    """train net"""
    set_parameter()
    print('device_id: {}, device_num: {}'.format(device_id, device_num),
          flush=True)
    ckpt_param_dict = load_pre_trained_checkpoint()
    dataset = create_dataset(dataset_path=config.dataset_path, do_train=True,
                             batch_size=config.batch_size, train_image_size=config.train_image_size,
                             distribute=config.run_distribute)
    step_size = dataset.get_dataset_size()
    net = resnet(class_num=config.class_num)
    init_weight(net=net, param_dict=ckpt_param_dict)
    lr = ms.Tensor(init_lr(step_size=step_size))
    # define opt
    group_params = init_group_params(net)
    opt = nn.Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    if config.optimizer == "LARS":
        opt = nn.LARS(opt, epsilon=config.lars_epsilon, coefficient=config.lars_coefficient,
                      lars_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'bias' not in x.name)
    loss = init_loss_scale()
    loss_scale = ms.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    dist_eval_network = ClassifyCorrectCell(net) if config.run_distribute else None
    metrics = {"acc"}
    if config.run_distribute:
        metrics = {'acc': DistAccuracy(batch_size=config.batch_size, device_num=device_num)}

    model = ms.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                     amp_level="O3", boost_level=config.boost_mode,
                     eval_network=dist_eval_network,
                     boost_config_dict={"grad_freeze": {"total_steps": config.epoch_size * step_size}})

    data_size = step_size * config.batch_size
    sync_frequency = 1
    federated_learning_manager = FederatedLearningManager(
        yaml_config=config.yaml_config,
        model=net,
        sync_frequency=sync_frequency,
        http_server_address=http_server_address,
        data_size=data_size,
        sync_type=config.sync_type,
        run_distribute=config.run_distribute,
        ssl_config=None
    )
    print('epoch: {}, step_size: {}, data_size: {}'.format(config.epoch_size, step_size, data_size),
          flush=True)
    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossCallBack(config.has_trained_epoch)
    cb = [time_cb, loss_cb, federated_learning_manager]
    # train model
    config.pretrain_epoch_size = config.has_trained_epoch
    model.train(config.epoch_size - config.pretrain_epoch_size, dataset, callbacks=cb,
                sink_size=step_size, dataset_sink_mode=True)


def start_one_server():
    """start one server"""
    from mindspore_federated import FLServerJob

    yaml_config = config.yaml_config
    tcp_server_ip = config.tcp_server_ip
    checkpoint_dir = config.checkpoint_dir

    job = FLServerJob(yaml_config=yaml_config, http_server_address=http_server_address, tcp_server_ip=tcp_server_ip,
                      checkpoint_dir=checkpoint_dir)
    job.run()


def start_one_scheduler():
    """start one scheduler"""
    from mindspore_federated import FlSchedulerJob

    yaml_config = config.yaml_config
    scheduler_manage_address = config.scheduler_manage_address

    job = FlSchedulerJob(yaml_config=yaml_config, manage_address=scheduler_manage_address)
    job.run()


if __name__ == '__main__':
    start = time.clock()
    ms_role = config.ms_role
    if ms_role == "MS_SERVER":
        start_one_server()
    elif ms_role == "MS_SCHED":
        start_one_scheduler()
    elif ms_role == "MS_WORKER":
        train_net()
    end = time.clock()
    print("total run time is:", (end - start), flush=True)
