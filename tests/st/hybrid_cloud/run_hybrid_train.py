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

import argparse
import os
import time
import ast
import numpy as np

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore import Tensor
import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as PV
import mindspore.dataset.transforms.py_transforms as PT
import mindspore.dataset.transforms.c_transforms as tC
from mindspore.train.serialization import save_checkpoint
from mindspore_federated.python._federated_local import FederatedLearningJob
from mindspore_federated.python.trainer._fl_manager import FederatedLearningManager, PushMetrics
from mindspore.nn.metrics import Accuracy
from mindspore.train import Model
from mindspore.train.callback import Callback

parser = argparse.ArgumentParser(description="test_cross_silo_femnist")
parser.add_argument("--device_target", type=str, default="CPU")
parser.add_argument("--server_mode", type=str, default="FEDERATED_LEARNING")
parser.add_argument("--ms_role", type=str, default="MS_WORKER")
parser.add_argument("--worker_num", type=int, default=0)
parser.add_argument("--server_num", type=int, default=1)
parser.add_argument("--scheduler_ip", type=str, default="127.0.0.1")
parser.add_argument("--scheduler_port", type=int, default=8113)
parser.add_argument("--fl_server_port", type=int, default=6666)
parser.add_argument("--start_fl_job_threshold", type=int, default=1)
parser.add_argument("--start_fl_job_time_window", type=int, default=3000)
parser.add_argument("--update_model_ratio", type=float, default=1.0)
parser.add_argument("--update_model_time_window", type=int, default=3000)
parser.add_argument("--fl_name", type=str, default="Lenet")
parser.add_argument("--fl_iteration_num", type=int, default=25)
parser.add_argument("--client_epoch_num", type=int, default=20)
parser.add_argument("--client_batch_size", type=int, default=32)
parser.add_argument("--client_learning_rate", type=float, default=0.1)
parser.add_argument("--scheduler_manage_port", type=int, default=11202)
parser.add_argument("--config_file_path", type=str, default="")
parser.add_argument("--pki_verify", type=ast.literal_eval, default=False)
# parameters used for pki_verify=True
parser.add_argument("--root_first_ca_path", type=str, default="")
parser.add_argument("--root_second_ca_path", type=str, default="")
parser.add_argument("--equip_crl_path", type=str, default="")
parser.add_argument("--replay_attack_time_diff", type=int, default=600000)
parser.add_argument("--encrypt_type", type=str, default="NOT_ENCRYPT")
# parameters for encrypt_type='DP_ENCRYPT'
parser.add_argument("--dp_eps", type=float, default=50.0)
parser.add_argument("--dp_delta", type=float, default=0.01)  # 1/worker_num
parser.add_argument("--dp_norm_clip", type=float, default=1.0)
# parameters for encrypt_type='PW_ENCRYPT'
parser.add_argument("--share_secrets_ratio", type=float, default=1.0)
parser.add_argument("--cipher_time_window", type=int, default=300000)
parser.add_argument("--reconstruct_secrets_threshold", type=int, default=3)
parser.add_argument("--client_password", type=str, default="")
parser.add_argument("--server_password", type=str, default="")
parser.add_argument("--enable_ssl", type=ast.literal_eval, default=False)
# parameters for 'SIGNDS'
parser.add_argument("--sign_k", type=float, default=0.01)
parser.add_argument("--sign_eps", type=float, default=100)
parser.add_argument("--sign_thr_ratio", type=float, default=0.6)
parser.add_argument("--sign_global_lr", type=float, default=0.1)
parser.add_argument("--sign_dim_out", type=int, default=0)
parser.add_argument("--global_iteration_time_window", type=int, default=3600000)
# parameters for "compression"
parser.add_argument("--upload_compress_type", type=str, default="NO_COMPRESS",
                    choices=["NO_COMPRESS", "DIFF_SPARSE_QUANT"])
parser.add_argument("--upload_sparse_rate", type=float, default=0.5)
parser.add_argument("--download_compress_type", type=str, default="NO_COMPRESS",
                    choices=["NO_COMPRESS", "QUANT"])
parser.add_argument("--checkpoint_dir", type=str, default="")
parser.add_argument("--model_path", type=str, default="")

args, _ = parser.parse_known_args()
device_target = args.device_target
server_mode = args.server_mode
ms_role = args.ms_role
worker_num = args.worker_num
server_num = args.server_num
scheduler_ip = args.scheduler_ip
scheduler_port = args.scheduler_port
fl_server_port = args.fl_server_port
start_fl_job_threshold = args.start_fl_job_threshold
start_fl_job_time_window = args.start_fl_job_time_window
update_model_ratio = args.update_model_ratio
update_model_time_window = args.update_model_time_window
fl_name = args.fl_name
fl_iteration_num = args.fl_iteration_num
client_epoch_num = args.client_epoch_num
client_batch_size = args.client_batch_size
client_learning_rate = args.client_learning_rate
scheduler_manage_port = args.scheduler_manage_port
config_file_path = args.config_file_path
encrypt_type = args.encrypt_type
share_secrets_ratio = args.share_secrets_ratio
cipher_time_window = args.cipher_time_window
reconstruct_secrets_threshold = args.reconstruct_secrets_threshold
dp_eps = args.dp_eps
dp_delta = args.dp_delta
dp_norm_clip = args.dp_norm_clip
client_password = args.client_password
server_password = args.server_password
enable_ssl = args.enable_ssl
pki_verify = args.pki_verify
root_first_ca_path = args.root_first_ca_path
root_second_ca_path = args.root_second_ca_path
equip_crl_path = args.equip_crl_path
replay_attack_time_diff = args.replay_attack_time_diff
sign_k = args.sign_k
sign_eps = args.sign_eps
sign_thr_ratio = args.sign_thr_ratio
sign_global_lr = args.sign_global_lr
sign_dim_out = args.sign_dim_out
global_iteration_time_window = args.global_iteration_time_window
upload_compress_type = args.upload_compress_type
upload_sparse_rate = args.upload_sparse_rate
download_compress_type = args.download_compress_type
checkpoint_dir = args.checkpoint_dir
model_path = args.model_path

ctx = {
    "server_mode": server_mode,
    "ms_role": ms_role,
    "worker_num": worker_num,
    "server_num": server_num,
    "scheduler_ip": scheduler_ip,
    "scheduler_port": scheduler_port,
    "fl_server_port": fl_server_port,
    "start_fl_job_threshold": start_fl_job_threshold,
    "start_fl_job_time_window": start_fl_job_time_window,
    "update_model_ratio": update_model_ratio,
    "update_model_time_window": update_model_time_window,
    "fl_name": fl_name,
    "fl_iteration_num": fl_iteration_num,
    "client_epoch_num": client_epoch_num,
    "client_batch_size": client_batch_size,
    "client_learning_rate": client_learning_rate,
    "scheduler_manage_port": scheduler_manage_port,
    "config_file_path": config_file_path,
    "pki_verify": pki_verify,
    "root_first_ca_path": root_first_ca_path,
    "root_second_ca_path": root_second_ca_path,
    "equip_crl_path": equip_crl_path,
    "replay_attack_time_diff": replay_attack_time_diff,
    "share_secrets_ratio": share_secrets_ratio,
    "cipher_time_window": cipher_time_window,
    "reconstruct_secrets_threshold": reconstruct_secrets_threshold,
    "dp_eps": dp_eps,
    "dp_delta": dp_delta,
    "dp_norm_clip": dp_norm_clip,
    "encrypt_type": encrypt_type,
    "client_password": client_password,
    "server_password": server_password,
    "enable_ssl": enable_ssl,
    "sign_k": sign_k,
    "sign_eps": sign_eps,
    "sign_thr_ratio": sign_thr_ratio,
    "sign_global_lr": sign_global_lr,
    "sign_dim_out": sign_dim_out,
    "global_iteration_time_window": global_iteration_time_window,
    "upload_compress_type": upload_compress_type,
    "upload_sparse_rate": upload_sparse_rate,
    "download_compress_type": download_compress_type,
    "checkpoint_dir": checkpoint_dir
}

context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
context.set_fl_context(**ctx)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_init=weight,
        has_bias=False,
        pad_mode="valid",
    )


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, channel=3):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv(channel, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class LossGet(Callback):
    # define loss callback for packaged model
    def __init__(self, per_print_times, data_size):
        super(LossGet, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self._loss = 0.0
        self.data_size = data_size
        self.loss_list = []

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training."
                             .format(cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            self._loss = loss
            self.loss_list.append(loss)

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        self._per_step_mseconds = epoch_mseconds / self.data_size

    def get_loss(self):
        return self.loss_list  # todo return self._loss

    def get_per_step_time(self):
        return self._per_step_mseconds


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def count_id(path):
    files = os.listdir(path)
    ids = {}
    for i in files:
        ids[i] = int(i)
    return ids


def create_dataset_from_folder(data_path, img_size, batch_size=32, repeat_size=1, num_parallel_workers=1,
                               shuffle=False):
    """ create dataset for train or test
        Args:
            data_path: Data path
            batch_size: The number of data records in each group
            repeat_size: The number of replicated data records
            num_parallel_workers: The number of parallel workers
        """
    # define dataset
    ids = count_id(data_path)
    mnist_ds = ds.ImageFolderDataset(dataset_dir=data_path, decode=False, class_indexing=ids)
    # define operation parameters
    resize_height, resize_width = img_size[0], img_size[1]

    transform = [
        PV.Decode(),
        PV.Grayscale(1),
        PV.Resize(size=(resize_height, resize_width)),
        PV.Grayscale(3),
        PV.ToTensor()
    ]
    compose = PT.Compose(transform)

    # apply map operations on images
    mnist_ds = mnist_ds.map(input_columns="label", operations=tC.TypeCast(mindspore.int32))
    mnist_ds = mnist_ds.map(input_columns="image", operations=compose)

    # apply DatasetOps
    buffer_size = 10000
    if shuffle:
        mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)
    return mnist_ds


def evalute_process(model, eval_data, img_size, batch_size):
    """Define the evaluation method."""
    ds_eval = create_dataset_from_folder(eval_data, img_size, batch_size)
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    return acc['Accuracy'], acc['Loss']


def train():
    epoch = 20
    network = LeNet5(62, 3)

    # construct dataset
    ds.config.set_seed(1)
    data_root_path = dataset_path
    user = "dataset_" + user_id
    train_path = os.path.join(data_root_path, user, "train")
    test_path = os.path.join(data_root_path, user, "test")

    dataset = create_dataset_from_folder(train_path, args.img_size, args.client_batch_size, args.repeat_size)
    print("size is ", dataset.get_dataset_size(), flush=True)
    num_batches = dataset.get_dataset_size()

    # define the loss function
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # define fl manager
    federated_learning_manager = FederatedLearningManager(
        server_mode=server_mode,
        model=network,
        sync_frequency=epoch * num_batches,
        data_size=num_batches,
        sync_type=sync_type,
        encrypt_type=encrypt_type
    )
    # define the optimizer
    net_opt = nn.Momentum(network.trainable_params(), client_learning_rate, 0.9)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy(), 'Loss': nn.Loss()})

    loss_cb = LossGet(1, num_batches)
    cbs = list()
    cbs.append(federated_learning_manager)
    cbs.append(loss_cb)
    ckpt_path = "ckpt"
    os.makedirs(ckpt_path)
    push_metrics = PushMetrics()
    for iter_num in range(fl_iteration_num):
        model.train(epoch, dataset, callbacks=cbs, dataset_sink_mode=False)

        ckpt_name = user_id + "-fl-ms-bs32-" + str(iter_num) + "epoch.ckpt"
        ckpt_name = os.path.join(ckpt_path, ckpt_name)
        save_checkpoint(network, ckpt_name)

        train_acc, _ = evalute_process(model, train_path, args.img_size, args.client_batch_size)
        test_acc, _ = evalute_process(model, test_path, args.img_size, args.client_batch_size)
        loss_list = loss_cb.get_loss()
        loss = sum(loss_list) / len(loss_list)
        print('local epoch: {}, loss: {}, trian acc: {}, test acc: {}'.format(iter_num, loss, train_acc, test_acc),
              flush=True)
        push_metrics.construct(loss, test_acc)

if __name__ == "__main__":
    fl_job = FederatedLearningJob(ctx)
    fl_job.run()
    if ms_role == "MS_WORKER":
        train()
