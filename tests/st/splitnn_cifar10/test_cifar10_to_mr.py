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
"""Test whether correctly convert cifar10 to mindrecord."""


import os
import sys
from cifar10_to_mr import SplitCifar10ToMR
import mindspore as ms
import mindspore.dataset as ds

if ms.__version__ <= "1.7.0":
    import mindspore.dataset.vision.c_transforms as vision
else:
    import mindspore.dataset.vision as vision


CIFAR10_PATH = os.path.join(sys.path[0], "datasets", "cifar-10-batches-py")
MINDRECORD_PATH = os.path.join(sys.path[0], "datasets", "mr_cifar10")


def generate_cifar10_mr():
    os.makedirs(MINDRECORD_PATH, exist_ok=True)
    cifar10_transformer = SplitCifar10ToMR(CIFAR10_PATH, os.path.join(MINDRECORD_PATH, "cifar10.mindrecord"))
    cifar10_transformer.transform(['label'])

def generate_cifar10_dataset(op, files):
    mds = ds.MindDataset(dataset_files=files)
    return mds.map(operations=op, input_columns=["data"], num_parallel_workers=2)

def load_cifar10_mr(file_part="cifar10.mindrecord_train_left"):
    ds.config.set_seed(0)
    decode_op = vision.Decode()
    file_path = os.path.join(MINDRECORD_PATH, file_part)
    data_set = generate_cifar10_dataset(decode_op, file_path)
    return data_set

def get_ids_and_labels(data_set):
    print("Got {} samples".format(data_set.get_dataset_size()))
    print("Got Column Names", data_set.get_col_names())
    ids, labels = [], []
    data_iter = data_set.create_tuple_iterator()
    for item in data_iter:
        ids.append(item[1])
        labels.append(item[2])
    return ids, labels

if __name__ == "__main__":
    generate_cifar10_mr()
    train_left = load_cifar10_mr("cifar10.mindrecord_train_left")
    train_right = load_cifar10_mr("cifar10.mindrecord_train_right")
    train_left_ids, train_left_labels = get_ids_and_labels(train_left)
    train_right_ids, train_right_labels = get_ids_and_labels(train_right)
    if train_left_ids == train_right_ids and train_left_labels == train_right_labels:
        print("The left and right part of training images are aligned.")

    test_left = load_cifar10_mr("cifar10.mindrecord_test_left")
    test_right = load_cifar10_mr("cifar10.mindrecord_test_right")
    test_left_ids, test_left_labels = get_ids_and_labels(test_left)
    test_right_ids, test_right_labels = get_ids_and_labels(test_right)
    if test_left_ids == test_right_ids and test_left_labels == test_right_labels:
        print("The left and right part of test images are aligned.")
