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
"""lenet network"""

import os
import time

import mindspore
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as tC
import mindspore.dataset.transforms.py_transforms as PT
import mindspore.dataset.vision.py_transforms as PV
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import TruncatedNormal
from mindspore.train.callback import Callback
import numpy as np


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
    """LeNet5"""
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
        # self.flatten = nn.Flatten()
        self.flatten = mindspore.ops.Flatten()

    def construct(self, x):
        """construct"""
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
    """# define loss callback for packaged network"""
    def __init__(self, per_print_times, data_size):
        super(LossGet, self).__init__()
        self._per_step_mseconds = None
        self.epoch_time = None
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self._loss = 0.0
        self.data_size = data_size
        self.loss_list = []

    def step_end(self, run_context):
        """step end"""
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

    def epoch_begin(self, _):
        """define epoch begin func"""
        self.epoch_time = time.time()

    def epoch_end(self, _):
        """define epoch end func"""
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        self._per_step_mseconds = epoch_mseconds / self.data_size

    def get_loss(self):
        """get loss"""
        return self.loss_list  # todo return self._loss

    def get_per_step_time(self):
        """get per step time"""
        return self._per_step_mseconds


def mkdir(path):
    """mkdir path"""
    if not os.path.exists(path):
        os.mkdir(path)


def count_id(path):
    """weight initial"""
    files = os.listdir(path)
    ids = {}
    for i in files:
        ids[i] = int(i)
    return ids


def create_dataset_from_folder(data_path, img_size, batch_size=32, repeat_size=1, shuffle=False):
    """ create dataset for train or test
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
        :param shuffle:
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
