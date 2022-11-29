# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""train FasterRcnn and get checkpoint files."""

import os
import sys
import time

import numpy as np
from mindspore import context, Tensor, Parameter
from mindspore.common import set_seed
from mindspore.common import dtype as mstype
from mindspore.nn import SGD
from mindspore.train import Model
from mindspore.train.callback import TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.serialization import save_checkpoint
from mindspore_federated import FederatedLearningManager

from src.FasterRcnn.faster_rcnn_resnet50v1 import Faster_Rcnn_Resnet
from src.dataset import create_fasterrcnn_dataset
from src.lr_schedule import dynamic_lr
from src.model_utils.config import config
from src.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

set_seed(1)

device_target = config.device_target
fl_iteration_num = config.fl_iteration_num
sync_type = config.sync_type
http_server_address = config.http_server_address
dataset_path = config.dataset_path
user_id = config.user_id

context.set_context(mode=context.GRAPH_MODE, device_target=device_target, device_id=int(config.device_id))
rank = 0
device_num = 1
user = "mindrecord_" + str(user_id)


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


def train_fasterrcnn_():
    """ train_fasterrcnn_ """
    print("Start create dataset!", flush=True)

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is FasterRcnn.mindrecord0, 1, ... file_num.
    prefix = "FasterRcnn.mindrecord"
    mindrecord_dir = config.dataset_path
    mindrecord_file = os.path.join(mindrecord_dir, user, prefix)
    print("CHECKING MINDRECORD FILES ...", mindrecord_file, flush=True)

    if rank == 0 and not os.path.exists(mindrecord_file):
        print("image_dir or anno_path not exits.", flush=True)

    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!", flush=True)

    # When create MindDataset, using the fitst mindrecord file, such as FasterRcnn.mindrecord0.
    dataset = create_fasterrcnn_dataset(config, mindrecord_file, batch_size=config.batch_size,
                                        device_num=device_num, rank_id=rank,
                                        num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing)

    num_batches = dataset.get_dataset_size()
    print("Create dataset done!", flush=True)

    return num_batches, dataset


def start_one_worker():
    """ train_fasterrcnn """
    num_batches, dataset = train_fasterrcnn_()
    net = Faster_Rcnn_Resnet(config=config)
    net = net.set_train()

    load_path = config.pre_trained
    print("Pre train path is:{}".format(load_path), flush=True)
    # load_path = ""
    if load_path != "":
        param_dict = load_checkpoint(load_path)

        key_mapping = {'down_sample_layer.1.beta': 'bn_down_sample.beta',
                       'down_sample_layer.1.gamma': 'bn_down_sample.gamma',
                       'down_sample_layer.0.weight': 'conv_down_sample.weight',
                       'down_sample_layer.1.moving_mean': 'bn_down_sample.moving_mean',
                       'down_sample_layer.1.moving_variance': 'bn_down_sample.moving_variance',
                       }
        for oldkey in list(param_dict.keys()):
            if not oldkey.startswith(('backbone', 'end_point', 'global_step', 'learning_rate', 'moments', 'momentum')):
                data = param_dict.pop(oldkey)
                newkey = 'backbone.' + oldkey
                param_dict[newkey] = data
                oldkey = newkey
            for k, v in key_mapping.items():
                if k in oldkey:
                    newkey = oldkey.replace(k, v)
                    param_dict[newkey] = param_dict.pop(oldkey)
                    break
        for item in list(param_dict.keys()):
            if not item.startswith('backbone'):
                param_dict.pop(item)

        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
        load_param_into_net(net, param_dict)
        print("load param dict size is:{}".format(len(net.trainable_params())), flush=True)

    loss = LossNet()
    lr = Tensor(dynamic_lr(config, num_batches), mstype.float32)
    sink_size = 27
    epoch = config.client_epoch_num
    data_size = num_batches * config.batch_size
    sync_frequency = epoch
    federated_learning_manager = FederatedLearningManager(
        yaml_config=config.yaml_config,
        model=net,
        sync_frequency=sync_frequency,
        http_server_address=http_server_address,
        data_size=data_size,
        sync_type=config.sync_type,
        ssl_config=None
    )
    print('epoch: {}, num_batches: {}, data_size: {}'.format(epoch, num_batches, data_size),
          flush=True)
    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    net_with_loss = WithLossCell(net, loss)
    if config.run_distribute:
        net = TrainOneStepCell(net_with_loss, opt, scale_sense=config.loss_scale)
    else:
        net = TrainOneStepCell(net_with_loss, opt, scale_sense=config.loss_scale)
    time_cb = TimeMonitor(data_size=num_batches)
    loss_cb = LossCallBack(rank_id=rank, lr=lr.asnumpy())
    cb = [federated_learning_manager, time_cb, loss_cb]

    model = Model(net)
    ckpt_path1 = os.path.join("ckpt", user)

    os.makedirs(ckpt_path1)
    print("====================", config.client_epoch_num, fl_iteration_num, flush=True)
    for iter_num in range(fl_iteration_num):
        model.train(config.client_epoch_num, dataset, callbacks=cb, dataset_sink_mode=True, sink_size=sink_size)
        ckpt_name = user + "-fast-rcnn-" + str(iter_num) + "-epoch.ckpt"
        ckpt_path = os.path.join(ckpt_path1, ckpt_name)
        save_checkpoint(net, ckpt_path)


if __name__ == "__main__":
    start = time.clock()
    ms_role = config.ms_role
    if ms_role == "MS_SERVER":
        start_one_server()
    elif ms_role == "MS_SCHED":
        start_one_scheduler()
    elif ms_role == "MS_WORKER":
        start_one_worker()
    end = time.clock()
    print("total run time is:", (end - start), flush=True)
