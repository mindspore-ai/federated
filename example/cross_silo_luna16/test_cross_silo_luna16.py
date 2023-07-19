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
"""start test cross silo luna16"""
import argparse
import numpy as np

from mindspore import set_seed
from mindspore_federated.startup.ssl_config import SSLConfig


set_seed(411)
parser = argparse.ArgumentParser(description="Run main_func case")

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
parser.add_argument("--data_dir", type=str, default="")
parser.add_argument("--sync_type", type=str, default="fixed", choices=["fixed", "adaptive"])
parser.add_argument("--client_batch_size", type=int, default=32)
parser.add_argument("--client_learning_rate", type=float, default=0.001)
parser.add_argument("--fl_iteration_num", type=int, default=25)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--local_epoch", type=int, default=20)
parser.add_argument("--ckpt_dir", type=str, default='./logs/worker_0/')

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
    from mindspore_federated.startup.federated_local import FLServerJob
    from src.unet import UNet

    yaml_config = args.yaml_config
    tcp_server_ip = args.tcp_server_ip
    http_server_address = args.http_server_address
    checkpoint_dir = args.checkpoint_dir

    job = FLServerJob(yaml_config=yaml_config, http_server_address=http_server_address, tcp_server_ip=tcp_server_ip,
                      checkpoint_dir=checkpoint_dir, ssl_config=ssl_config)

    # server need init parameters of network to be started
    network = UNet()
    feature_map = get_trainable_params(network)
    job.run(feature_map)


def start_one_scheduler():
    """start one scheduler"""
    from mindspore_federated import FlSchedulerJob

    yaml_config = args.yaml_config
    scheduler_manage_address = args.scheduler_manage_address

    # connect to redis as a Distributed cache
    job = FlSchedulerJob(yaml_config=yaml_config, manage_address=scheduler_manage_address)
    job.run()


def start_one_worker():
    """start one start_one_worker"""
    from mindspore import nn, context
    from mindspore_federated.trainer._fl_manager import FederatedLearningFedCMManager
    from mindspore_federated.trainer.hfl_model import FedCMModel
    from src.unet import UNet
    from src.dataset import load_bin_dataset
    from src.loss import CrossEntropyWithLogits
    from src.utils import LossGet

    yaml_config = args.yaml_config
    sync_type = args.sync_type
    client_learning_rate = args.client_learning_rate
    fl_iteration_num = args.fl_iteration_num
    client_batch_size = args.client_batch_size
    device_target = args.device_target
    http_server_address = args.http_server_address
    device_id = args.device_id
    data_dir = args.data_dir
    epoch = args.local_epoch
    network = UNet()
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target, device_id=device_id)

    # construct dataset
    dataset = load_bin_dataset(
        data_dir=data_dir,
        feature_shape=(-1, 1, 256, 256),
        label_shape=(-1, 1, 256, 256),
        batch_size=20,
        do_shuffle=True,
        drop_remainder=False,
        feature_dtype='float32',
        label_dtype='float32',
    )
    num_batches = dataset.get_dataset_size()
    print("size is ", num_batches)
    data_size = num_batches * client_batch_size

    # define the loss function
    net_loss = CrossEntropyWithLogits()
    loss_callback = LossGet(per_print_times=1, data_size=data_size)

    # define the optimizer
    net_opt = nn.Adam(network.trainable_params(), client_learning_rate)

    # define high level mindspore federated api
    model = FedCMModel(network, net_loss, net_opt, loss_callback)

    # define mindspore federated worker communicator manager
    sync_frequency = epoch * num_batches
    federated_learning_manager = FederatedLearningFedCMManager(
        yaml_config=yaml_config,
        model=network,
        fedcm_model=model,
        sync_frequency=sync_frequency,
        http_server_address=http_server_address,
        data_size=data_size,
        sync_type=sync_type,
        ssl_config=ssl_config,
        delta_p=client_learning_rate * num_batches * epoch
    )
    print('epoch: {}, num_batches: {}, data_size: {}'.format(epoch, num_batches, data_size))

    # define loss callback

    cbs = [federated_learning_manager, loss_callback]

    # main loop
    for iter_num in range(fl_iteration_num):
        model.train(epoch, dataset, callbacks=cbs, dataset_sink_mode=False)
        loss_list = loss_callback.get_loss()
        preloss_list = loss_callback.get_preloss()
        step_time = loss_callback.get_per_step_time()
        print('iteration: {}, \
              loss: {}, \
              step_time: {}, \
              length: {}'.format(iter_num, np.average(loss_list), step_time, len(loss_list)))
        print('iteration: {}, \
              preloss: {}, \
              step_time: {}, \
              length: {}'.format(iter_num, np.average(preloss_list), step_time, len(preloss_list)))

if __name__ == "__main__":
    ms_role = args.ms_role
    if ms_role == "MS_SERVER":
        start_one_server()
    elif ms_role == "MS_SCHED":
        start_one_scheduler()
    elif ms_role == "MS_WORKER":
        start_one_worker()
