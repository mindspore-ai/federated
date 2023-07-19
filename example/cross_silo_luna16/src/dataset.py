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
"""start get cross silo dataset"""

import os
import numpy as np
from mindspore import dataset as ds


def load_bin_dataset(data_dir, feature_shape=(-1, 3, 24, 24), label_shape=(-1,),
                     batch_size=20, do_shuffle=True, drop_remainder=False,
                     feature_dtype='float32', label_dtype='int32'):
    """load dataset"""
    dtype_dict = {
        'uint8': np.uint8,
        'int8': np.int8,
        'int16': np.int16,
        'int32': np.int32,
        'int64': np.int64,
        'float16': np.float16,
        'float32': np.float32,
        'float64': np.float64,
    }
    if feature_dtype in dtype_dict:
        feature_dtype = dtype_dict[feature_dtype]
    else:
        raise ValueError("feature_dtype must in {}".format(dtype_dict.keys()))
    if label_dtype in dtype_dict:
        label_dtype = dtype_dict[label_dtype]
    else:
        raise ValueError("label_dtype must in {}".format(dtype_dict.keys()))

    feature_path = os.path.join(data_dir, 'feature.bin')
    label_path = os.path.join(data_dir, 'label.bin')

    feature_np = np.fromfile(feature_path, dtype=feature_dtype).reshape(feature_shape)
    label_np = np.fromfile(label_path, dtype=label_dtype).reshape(label_shape)

    def generator_func():
        for feature, label in zip(feature_np, label_np):
            yield np.array(feature), np.array(label)

    dataset = ds.GeneratorDataset(generator_func, ['feature', 'label'], num_parallel_workers=8)
    if do_shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    return dataset
