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
"""Essential tools to modeling the split-learning process."""

import base64
from typing import OrderedDict
import numpy as np

from mindspore import ops, Tensor
from mindspore_federated._mindspore_federated import TensorListItem_, TensorItem_

DATATYPE_STRING_NPTYPE_DICT = {
    "float32": np.float32,
    "uint8": np.uint8,
    "int8": np.int8,
    "uint16": np.uint16,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
    "float16": np.float16,
    "double": np.double,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float64": np.float64
}


def tensor_to_tensor_pybind_obj(ts: Tensor, ref_key: str = None):
    """
    Convert a tensor to a pybind item.
    Inputs:
        ts (class): the Tensor object.
        ref_key (str): the ref name of the Tensor object.
    """
    if ts is None:
        raise ValueError('tensor_to_tensor_pybind_obj: could not input a Tensor with None value')
    tensor = TensorItem_()
    if ref_key:
        tensor.set_ref_key(ref_key)
    if ts.dtype is None:
        raise TypeError('tensor_to_tensor_pybind_obj: input a Tensor with unsupported value type.')
    tensor.set_shape(ts.shape)
    np_data = ts.reshape(ts.size,).asnumpy()
    np_data_b64 = base64.b64encode(np_data.tobytes())
    tensor.set_raw_data(np_data_b64)
    tensor.set_dtype(str(np_data.dtype))
    return tensor


def tensor_dict_to_tensor_list_pybind_obj(ts_dict: dict, name: str = 'no_name', send_addr: str = 'localhost',
                                          recv_addr: str = 'localhost'):
    """
    Convert a dict, the s of which are tensor objects, to pybind object.
    Inputs:
        ts_dict (dict): the dict object, the items of which are tensors.
        name (str): the ref name of the pybind object.
        send_addr (str): the send address of the pybind object.
        recv_addr (str): the receive address of the pybind object.
    """
    tensor_list_item = TensorListItem_()
    tensor_list_item.set_name(name)
    for ts_key, ts in ts_dict.items():
        if isinstance(ts, Tensor):
            tensor = tensor_to_tensor_pybind_obj(ts, ts_key)
            tensor_list_item.add_tensor(tensor)
        elif isinstance(ts, OrderedDict):
            sub_tensor_list_item = tensor_dict_to_tensor_list_pybind_obj(ts, ts_key, send_addr, recv_addr)
            tensor_list_item.add_tensor_list_item(sub_tensor_list_item)
    return tensor_list_item


def tensor_pybind_obj_to_tensor(tensor_item: TensorItem_):
    """
    Convert a pybind to a tensor.
    Inputs:
        tensor_item (class): the pybind object of the Tensor.
    """
    ref_key = tensor_item.ref_key()
    dtype_str = tensor_item.dtype()
    shape = tensor_item.shape()
    if dtype_str is None or dtype_str == "":
        raise ValueError('tensor_pybind_obj_to_tensor: Tensor with unsupported value type.')
    dtype = DATATYPE_STRING_NPTYPE_DICT.get(dtype_str)
    values = base64.b64decode(tensor_item.raw_data())
    np_data = np.frombuffer(values, dtype=dtype).reshape(shape)

    ts = Tensor(np_data)
    ts = ops.reshape(ts, tuple(shape))
    return ref_key, ts


def tensor_list_pybind_obj_to_tensor_dict(tensor_list_item: TensorListItem_):
    """
    Parse a dict, the s of which are tensor objects, from a pybind object.
    Inputs:
        tensor_list_item (class): the pybind object.
    """
    name = tensor_list_item.name()
    res = OrderedDict()
    tensors = list(tensor_list_item.tensors())
    for ts_item in tensors:
        ref_key, ts = tensor_pybind_obj_to_tensor(ts_item)
        res[ref_key] = ts
    tensor_dicts = list(tensor_list_item.tensorListItems())
    for ts_list_item in tensor_dicts:
        sub_name, sub_res = tensor_list_pybind_obj_to_tensor_dict(ts_list_item)
        res[sub_name] = sub_res
    return name, res
