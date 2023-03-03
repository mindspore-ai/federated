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
"""SSL config for start up FL"""
import numpy as np


def gen_data_for_vae():
    """
    generate ut test data for vae model.
    """
    # input_data = np.random.random((32, 1039)).astype(np.float32)
    input_data = np.random.random((16, 1039)).astype(np.float32)
    input_data.tofile("input1.bin")

    # input_data = np.random.random((32, 64)).astype(np.float32)
    input_data = np.random.random((16, 64)).astype(np.float32)
    input_data.tofile("input2.bin")


def gen_data_for_lenet():
    """
    generate ut test data for lenet model.
    """
    # input_data = np.random.random((32, 32, 32, 3)).astype(np.float32)
    input_data = np.random.random((512, 32, 32, 3)).astype(np.float32)
    input_data.tofile("test_data/lenet/f0178_39/f0178_39_bn_9_train_data.bin")

    # input_data = np.random.random((32, 62)).astype(np.float32)
    input_data = np.random.random_integers(0, 61, 512).astype(np.int32)
    input_data.tofile("test_data/lenet/f0178_39/f0178_39_bn_9_train_label.bin")

    # input_data = np.random.random((32, 32, 32, 3)).astype(np.float32)
    input_data = np.random.random((128, 32, 32, 3)).astype(np.float32)
    input_data.tofile("test_data/lenet/f0178_39/f0178_39_bn_1_test_data.bin")

    # input_data = np.random.random((32, 62)).astype(np.float32)
    input_data = np.random.random_integers(0, 61, 128).astype(np.int32)
    input_data.tofile("test_data/lenet/f0178_39/f0178_39_bn_1_test_label.bin")


gen_data_for_lenet()
