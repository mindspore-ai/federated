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
"""start running lenet network of cross silo mode"""
import argparse
import ast
import os
import sys
import time

import numpy as np
import mindspore as ms
from mindspore.communication.management import get_rank
from mindspore_federated.startup.ssl_config import SSLConfig
from mindspore_federated.trainer.hfl_model import HFLModel

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

parser = argparse.ArgumentParser(description="Run test_cross_silo_femnist.py case")

parser.add_argument("--ms_role", type=str, default="MS_WORKER")
# common
parser.add_argument("--yaml_config", type=str, default="default_yaml_config.yaml")
# for server
parser.add_argument("--http_server_address", type=str, default="127.0.0.1:5555")
parser.add_argument("--tcp_server_ip", type=str, default="127.0.0.1")
parser.add_argument("--checkpoint_dir", type=str, default="./fl_ckpt/")

# for scheduler
parser.add_argument("--scheduler_manage_address", type=str, default="127.0.0.1:11202")

# for worker
parser.add_argument("--device_target", type=str, default="GPU")
parser.add_argument("--dataset_path", type=str, default="")
# The user_id is used to set each worker's dataset path.
parser.add_argument("--user_id", type=str, default="0")
parser.add_argument("--sync_type", type=str, default="fixed", choices=["fixed", "adaptive"])

parser.add_argument('--img_size', type=int, default=(32, 32, 1), help='the image size of (h,w,c)')
parser.add_argument('--repeat_size', type=int, default=1, help='the repeat size when create the dataLoader')

# client_batch_size is also used as the batch size of each mini-batch for Worker.
parser.add_argument("--client_batch_size", type=int, default=32)
parser.add_argument("--client_learning_rate", type=float, default=0.01)
parser.add_argument("--fl_iteration_num", type=int, default=25)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--device_num", type=int, default=1)
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False)

args, _ = parser.parse_known_args()

ssl_config = SSLConfig(server_password="server_password_12345", client_password="client_password_12345")


def get_trainable_params(network):
    """get trainable params"""
    feature_map = {}
    for param in network.trainable_params():
        param_np = param.asnumpy()
        if param_np.dtype != np.float32:
            continue
        feature_map[param.name] = param_np
    return feature_map


def start_one_server():
    """start one server"""
    from network.lenet import LeNet5
    from mindspore_federated import FLServerJob

    yaml_config = args.yaml_config
    tcp_server_ip = args.tcp_server_ip
    http_server_address = args.http_server_address
    checkpoint_dir = args.checkpoint_dir

    network = LeNet5(62, 3)
    job = FLServerJob(yaml_config=yaml_config, http_server_address=http_server_address, tcp_server_ip=tcp_server_ip,
                      checkpoint_dir=checkpoint_dir, ssl_config=ssl_config)
    feature_map = get_trainable_params(network)
    job.run(feature_map)


def start_one_scheduler():
    """start one scheduler"""
    from mindspore_federated import FlSchedulerJob

    yaml_config = args.yaml_config
    scheduler_manage_address = args.scheduler_manage_address

    job = FlSchedulerJob(yaml_config=yaml_config, manage_address=scheduler_manage_address, ssl_config=ssl_config)
    job.run()


def start_one_worker():
    """start one start_one_worker"""
    yaml_config = args.yaml_config
    dataset_path = args.dataset_path
    user_id = args.user_id
    sync_type = args.sync_type
    client_learning_rate = args.client_learning_rate
    fl_iteration_num = args.fl_iteration_num
    client_batch_size = args.client_batch_size
    img_size = args.img_size
    repeat_size = args.repeat_size
    device_target = args.device_target
    http_server_address = args.http_server_address
    device_id = args.device_id
    run_distribute = args.run_distribute
    device_num = args.device_num

    from mindspore import nn, save_checkpoint, context
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target, device_id=device_id)

    from mindspore_federated import FederatedLearningManager
    from mindspore.communication.management import init
    from network.lenet import LeNet5, ds, create_dataset_from_folder, LossGet, evalute_process
    epoch = 20
    network = LeNet5(62, 3)
    from mindspore.nn.metrics import Accuracy
    print("device_id is {}".format(device_id))
    rank_id = 0
    if run_distribute:
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
        rank_id = get_rank()
    # construct dataset
    ds.config.set_seed(1)
    data_root_path = dataset_path
    user = "dataset_" + user_id
    train_path = os.path.join(data_root_path, user, "train")
    test_path = os.path.join(data_root_path, user, "test")

    dataset = create_dataset_from_folder(train_path, img_size, client_batch_size, repeat_size)
    print("size is ", dataset.get_dataset_size(), flush=True)
    num_batches = dataset.get_dataset_size()

    # define the loss function
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # define fl manager

    data_size = num_batches * client_batch_size
    sync_frequency = epoch * num_batches
    federated_learning_manager = FederatedLearningManager(
        yaml_config=yaml_config,
        model=network,
        sync_frequency=sync_frequency,
        http_server_address=http_server_address,
        data_size=data_size,
        sync_type=sync_type,
        ssl_config=ssl_config,
        run_distribute=run_distribute
    )
    print('epoch: {}, num_batches: {}, data_size: {}'.format(epoch, num_batches, data_size),
          flush=True)
    # define the optimizer
    net_opt = nn.Momentum(network.trainable_params(), client_learning_rate, 0.9)
    model = HFLModel(network, net_loss, net_opt, metrics={"Accuracy": Accuracy(), 'Loss': nn.Loss()})

    loss_cb = LossGet(1, num_batches)
    cbs = list()
    cbs.append(federated_learning_manager)
    cbs.append(loss_cb)
    ckpt_dir = "ckpt"
    if rank_id == 0:
        os.makedirs(ckpt_dir)

    for iter_num in range(fl_iteration_num):
        model.train(epoch, dataset, callbacks=cbs, dataset_sink_mode=False)
        ckpt_name = user_id + "-fl-ms-bs32-" + str(iter_num) + "epoch.ckpt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if rank_id == 0:
            save_checkpoint(network, ckpt_path)

        train_acc, _ = evalute_process(model, train_path, img_size, client_batch_size)
        test_acc, _ = evalute_process(model, test_path, img_size, client_batch_size)
        loss_list = loss_cb.get_loss()
        loss = sum(loss_list) / len(loss_list)
        print('local epoch: {}, loss: {}, train acc: {}, test acc: {}'.format(iter_num, loss, train_acc, test_acc),
              flush=True)


if __name__ == "__main__":
    start = time.clock()
    ms_role = args.ms_role
    if ms_role == "MS_SERVER":
        start_one_server()
    elif ms_role == "MS_SCHED":
        start_one_scheduler()
    elif ms_role == "MS_WORKER":
        start_one_worker()
    end = time.clock()
    print("total run time is:", (end - start), flush=True)
