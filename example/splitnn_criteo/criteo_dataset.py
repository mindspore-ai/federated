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
"""This module provides methods and classes reading the preprocessed criteo dataset."""

import os
from enum import Enum

import numpy as np

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication import get_rank, get_group_size
import mindspore.dataset as ds
import mindspore.common.dtype as mstype
from mindspore import version
from mindspore_federated.data_join import load_mindrecord


class DataType(Enum):
    """
    Enumerate supported dataset format.
    """
    MINDRECORD = 1
    TFRECORD = 2
    H5 = 3


def _padding_func(batch_size, manual_shape, target_column, cut_point, field_size=39):
    """
    get padding_func
    """
    if manual_shape:
        generate_concat_offset = [item[0] + item[1] for item in manual_shape]
        part_size = int(target_column / len(generate_concat_offset))
        filled_value = []
        for i in range(field_size, target_column):
            filled_value.append(generate_concat_offset[i // part_size] - 1)
        print("Filed Value:", filled_value)

        def padding_func(x, y, z):
            x = np.array(x).flatten().reshape(batch_size, field_size)
            y = np.array(y).flatten().reshape(batch_size, field_size)
            z = np.array(z).flatten().reshape(batch_size, 1)

            x_id = np.ones((batch_size, target_column - field_size), dtype=np.int32) * filled_value
            x_id = np.concatenate([x, x_id.astype(dtype=np.int32)], axis=1)
            mask = np.concatenate([y, np.zeros((batch_size, target_column - 39), dtype=np.float32)], axis=1)
            return (x_id, mask, z)
    else:
        def padding_func(x, y, z):
            x = np.array(x).flatten().reshape(batch_size, field_size)
            mask0 = list(range(cut_point[0])) + list(range(cut_point[1], cut_point[2]))
            mask1 = list(range(cut_point[0], cut_point[1])) + list(range(cut_point[2], field_size))
            x0, x1 = x[:, mask0], x[:, mask1]
            y = np.array(y).flatten().reshape(batch_size, field_size)
            y0, y1 = y[:, mask0], y[:, mask1]
            z = np.array(z).flatten().reshape(batch_size, 1)
            return x0, x1, y0, y1, z
    return padding_func


def _get_tf_dataset(data_dir, train_mode=True, batch_size=1000,
                    line_per_sample=1000, rank_size=None, rank_id=None,
                    manual_shape=None, target_column=40):
    """
    get_tf_dataset
    """
    dataset_files = []
    file_prefix_name = 'train' if train_mode else 'test'
    shuffle = train_mode
    for (dirpath, _, filenames) in os.walk(data_dir):
        for filename in filenames:
            if file_prefix_name in filename and "tfrecord" in filename:
                dataset_files.append(os.path.join(dirpath, filename))
    schema = ds.Schema()
    schema.add_column('feat_ids', de_type=mstype.int32)
    schema.add_column('feat_vals', de_type=mstype.float32)
    schema.add_column('label', de_type=mstype.float32)
    if rank_size is not None and rank_id is not None:
        data_set = ds.TFRecordDataset(dataset_files=dataset_files, shuffle=shuffle, schema=schema,
                                      num_parallel_workers=8,
                                      num_shards=rank_size, shard_id=rank_id, shard_equal_rows=True)
    else:
        data_set = ds.TFRecordDataset(dataset_files=dataset_files,
                                      shuffle=shuffle, schema=schema, num_parallel_workers=8)
    data_set = data_set.batch(int(batch_size / line_per_sample), drop_remainder=True)

    if version.__version__.startswith("2."):
        data_set = data_set.map(
            operations=_padding_func(batch_size, manual_shape, target_column, cut_point=[7, 13, 26]),
            input_columns=['feat_ids', 'feat_vals', 'label'],
            num_parallel_workers=8)
        data_set = data_set.project(['feat_ids', 'feat_vals', 'label'])
    else:
        data_set = data_set.map(
            operations=_padding_func(batch_size, manual_shape, target_column, cut_point=[7, 13, 26]),
            input_columns=['feat_ids', 'feat_vals', 'label'],
            column_order=['feat_ids', 'feat_vals', 'label'], num_parallel_workers=8)
    return data_set


def _get_mindrecord_dataset(directory, train_mode=True, batch_size=1000,
                            line_per_sample=1000, rank_size=None, rank_id=None,
                            manual_shape=None, target_column=40):
    """
    Get dataset with mindrecord format.

    Args:
        directory (str): Dataset directory.
        train_mode (bool): Whether dataset is use for train or eval (default=True).
        batch_size (int): Dataset batch size (default=1000).
        line_per_sample (int): The number of sample per line (default=1000).
        rank_size (int): The number of device, not necessary for single device (default=None).
        rank_id (int): Id of device, not necessary for single device (default=None).

    Returns:
        Dataset.
    """
    file_prefix_name = 'train_input_part.mindrecord' if train_mode else 'test_input_part.mindrecord'
    file_suffix_name = '00' if train_mode else '0'
    shuffle = train_mode

    if rank_size is not None and rank_id is not None:
        data_set = ds.MindDataset(os.path.join(directory, file_prefix_name + file_suffix_name),
                                  columns_list=['feat_ids', 'feat_vals', 'label'],
                                  num_shards=rank_size, shard_id=rank_id, shuffle=shuffle,
                                  num_parallel_workers=8)
    else:
        data_set = ds.MindDataset(os.path.join(directory, file_prefix_name + file_suffix_name),
                                  columns_list=['feat_ids', 'feat_vals', 'label'],
                                  shuffle=shuffle, num_parallel_workers=8)
    data_set = data_set.batch(int(batch_size / line_per_sample), drop_remainder=True)
    if version.__version__.startswith("2."):
        data_set = data_set.map(_padding_func(batch_size, manual_shape, target_column, cut_point=[7, 13, 26]),
                                input_columns=['feat_ids', 'feat_vals', 'label'],
                                output_columns=['id_hldr', 'id_hldr0', 'wt_hldr', 'wt_hldr0', 'label'],
                                num_parallel_workers=8)
        data_set = data_set.project(['id_hldr', 'id_hldr0', 'wt_hldr', 'wt_hldr0', 'label'])
    else:
        data_set = data_set.map(_padding_func(batch_size, manual_shape, target_column, cut_point=[7, 13, 26]),
                                input_columns=['feat_ids', 'feat_vals', 'label'],
                                output_columns=['id_hldr', 'id_hldr0', 'wt_hldr', 'wt_hldr0', 'label'],
                                column_order=['id_hldr', 'id_hldr0', 'wt_hldr', 'wt_hldr0', 'label'],
                                num_parallel_workers=8)
    return data_set


def create_dataset(data_dir, train_mode=True, batch_size=1000,
                   data_type=DataType.TFRECORD, line_per_sample=1000,
                   rank_size=None, rank_id=None, manual_shape=None, target_column=40):
    """
    create dataset object from file folder

    Args:
        data_dir (str): data file dictionary
        train_mode (bool): whether dataset is use for train or eval (default=True).
        batch_size (int): dataset batch size (default=1000).
        data_type (Enum): type of data file (default=TFRECORD), MINDRECORD, TFRECORD, or H5.
        rank_size (int): the number of device, not necessary for single device (default=None).
        rank_id (int): ID of device, not necessary for single device (default=None).

    Return:
        Dataset object
    """
    if data_type not in DataType:
        raise ValueError("Unsupported data_type")
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == ParallelMode.DATA_PARALLEL:
        rank_id = get_rank()
        rank_size = get_group_size()
    if data_type == DataType.TFRECORD:
        return _get_tf_dataset(data_dir, train_mode, batch_size,
                               line_per_sample, rank_size=rank_size, rank_id=rank_id,
                               manual_shape=manual_shape, target_column=target_column)
    if data_type == DataType.MINDRECORD:
        return _get_mindrecord_dataset(data_dir, train_mode, batch_size,
                                       line_per_sample, rank_size=rank_size, rank_id=rank_id,
                                       manual_shape=manual_shape, target_column=target_column)
    raise RuntimeError("Please use TFRECORD or MINDRECORD dataset in parallel mode.")


def combine_leader(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19,
        y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19,
        label
):
    """combine leader data"""
    x = x0.flatten()
    x = np.append(x, x1.flatten())
    x = np.append(x, x2.flatten())
    x = np.append(x, x3.flatten())
    x = np.append(x, x4.flatten())
    x = np.append(x, x5.flatten())
    x = np.append(x, x6.flatten())
    x = np.append(x, x7.flatten())
    x = np.append(x, x8.flatten())
    x = np.append(x, x9.flatten())
    x = np.append(x, x10.flatten())
    x = np.append(x, x11.flatten())
    x = np.append(x, x12.flatten())
    x = np.append(x, x13.flatten())
    x = np.append(x, x14.flatten())
    x = np.append(x, x15.flatten())
    x = np.append(x, x16.flatten())
    x = np.append(x, x17.flatten())
    x = np.append(x, x18.flatten())
    x = np.append(x, x19.flatten())

    y = y0.flatten()
    y = np.append(y, y1.flatten())
    y = np.append(y, y2.flatten())
    y = np.append(y, y3.flatten())
    y = np.append(y, y4.flatten())
    y = np.append(y, y5.flatten())
    y = np.append(y, y6.flatten())
    y = np.append(y, y7.flatten())
    y = np.append(y, y8.flatten())
    y = np.append(y, y9.flatten())
    y = np.append(y, y10.flatten())
    y = np.append(y, y11.flatten())
    y = np.append(y, y12.flatten())
    y = np.append(y, y13.flatten())
    y = np.append(y, y14.flatten())
    y = np.append(y, y15.flatten())
    y = np.append(y, y16.flatten())
    y = np.append(y, y17.flatten())
    y = np.append(y, y18.flatten())
    y = np.append(y, y19.flatten())
    return x, y, label.reshape((-1,))


def combine_follower(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18,
        y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18,
):
    """combine follower data"""
    x = x0.flatten()
    x = np.append(x, x1.flatten())
    x = np.append(x, x2.flatten())
    x = np.append(x, x3.flatten())
    x = np.append(x, x4.flatten())
    x = np.append(x, x5.flatten())
    x = np.append(x, x6.flatten())
    x = np.append(x, x7.flatten())
    x = np.append(x, x8.flatten())
    x = np.append(x, x9.flatten())
    x = np.append(x, x10.flatten())
    x = np.append(x, x11.flatten())
    x = np.append(x, x12.flatten())
    x = np.append(x, x13.flatten())
    x = np.append(x, x14.flatten())
    x = np.append(x, x15.flatten())
    x = np.append(x, x16.flatten())
    x = np.append(x, x17.flatten())
    x = np.append(x, x18.flatten())

    y = y0.flatten()
    y = np.append(y, y1.flatten())
    y = np.append(y, y2.flatten())
    y = np.append(y, y3.flatten())
    y = np.append(y, y4.flatten())
    y = np.append(y, y5.flatten())
    y = np.append(y, y6.flatten())
    y = np.append(y, y7.flatten())
    y = np.append(y, y8.flatten())
    y = np.append(y, y9.flatten())
    y = np.append(y, y10.flatten())
    y = np.append(y, y11.flatten())
    y = np.append(y, y12.flatten())
    y = np.append(y, y13.flatten())
    y = np.append(y, y14.flatten())
    y = np.append(y, y15.flatten())
    y = np.append(y, y16.flatten())
    y = np.append(y, y17.flatten())
    y = np.append(y, y18.flatten())
    return x, y


def create_joined_dataset(dataset_dir, train_mode=True, batch_size=1000, seed=0, drop_remainder=True, role="leader"):
    """
    create dataset object from file folder

    Args:
        dataset_dir (str): data file dictionary
        train_mode (bool): whether dataset is use for train or eval (default=True).
        batch_size (int): dataset batch size (default=1000).
        seed (int): random shuffle seed (default=0).
        drop_remainder (bool): drop or not (default=True).
        role (str): role of current data (default="leader").

    Return:
        Dataset object
    """
    dataset = load_mindrecord(input_dir=dataset_dir, shuffle=train_mode, seed=seed, num_parallel_workers=8)

    if role == "leader":
        names = list(str(_) for _ in range(7)) + list(str(_) for _ in range(13, 26))
        ids_name = "id_hldr"
        vals_name = "wt_hldr"
        input_columns = ["feat_ids_" + name for name in names] + ["feat_vals_" + name for name in names] + ["label"]
        output_columns = [ids_name, vals_name, "label"]
        combine = combine_leader
    else:
        names = list(str(_) for _ in range(7, 13)) + list(str(_) for _ in range(26, 39))
        ids_name = "id_hldr0"
        vals_name = "wt_hldr0"
        input_columns = ["feat_ids_" + name for name in names] + ["feat_vals_" + name for name in names]
        output_columns = [ids_name, vals_name]
        combine = combine_follower
    if version.__version__.startswith("2."):
        dataset = dataset.map(
            operations=combine,
            input_columns=input_columns,
            output_columns=output_columns
        )
        dataset = dataset.project(output_columns)
    else:
        dataset = dataset.map(
            operations=combine,
            input_columns=input_columns,
            column_order=output_columns,
            output_columns=output_columns
        )
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset
