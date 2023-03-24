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
"""Export and load data in vfl."""

import os
from joblib import Parallel, delayed
from mindspore.mindrecord import FileWriter
from mindspore import dataset as ds


def _split_keys(keys, shard_num):
    """
    Split keys.
    """
    keys_list = list()
    keys_len = len(keys)
    before_len = keys_len // shard_num
    after_len = before_len + 1
    change_index = shard_num - keys_len % shard_num - 1
    start_index = 0
    length = before_len
    for shard_index in range(shard_num):
        keys_list.append(keys[start_index:(start_index+length)])
        start_index += length
        length = before_len if shard_index < change_index else after_len
    return keys_list


def export_mindrecord(file_name, raw_data, keys, shard_num=1, overwrite=True):
    """
    Export mindrecord.
    """
    schema = raw_data.schema()
    desc = raw_data.desc()
    if shard_num == 1:
        export_single_mindrecord(file_name, raw_data, keys, schema, desc, overwrite)
    else:
        export_multi_mindrecord(file_name, raw_data, keys, schema, desc, shard_num, overwrite)


def export_single_mindrecord(file_name, raw_data, keys, schema, desc, overwrite=True):
    """
    Export single mindrecord.
    """
    if not keys:
        return
    writer = FileWriter(file_name=file_name, shard_num=1, overwrite=overwrite)
    writer.add_schema(schema, desc)
    writer.open_and_set_header()
    for value in raw_data.values(keys=keys):
        writer.write_raw_data([value])
    writer.commit()


def export_multi_mindrecord(file_name, raw_data, keys, schema, desc, shard_num, overwrite=True):
    """
    Export multi mindrecord.
    """
    keys_list = _split_keys(keys, shard_num)
    Parallel(n_jobs=shard_num)(
        delayed(export_single_mindrecord)(file_name + str(shard_id), raw_data, keys, schema, desc, overwrite)
        for shard_id, keys in enumerate(keys_list)
    )


def load_mindrecord(input_dir, seed=0, **kwargs):
    """
    Load MindRecord files.

    Args:
        input_dir (str): Input directory for storing MindRecord-related files.
        seed (int): The random seed. Default: 0.

    Returns:
        MindDataset, Order-preserving datasets.

    Note:
        This API transparently transfers the `kwargs` to MindDataset.
        For details about more hyper parameters in `kwargs`, refer to `mindspore.dataset.MindDataset` .

    Examples:
        >>> dataset = load_mindrecord(input_dir="input_dir", seed=0, shuffle=True)
        >>> for batch in dataset.create_tuple_iterator():
        ...     print(batch)
    """
    ds.config.set_seed(seed)
    dataset_files = [os.path.join(input_dir, _) for _ in os.listdir(input_dir) if "db" not in _]
    dataset_files.sort()
    dataset = ds.MindDataset(dataset_files, **kwargs)
    return dataset
