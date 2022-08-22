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

import logging
from typing import OrderedDict

from mindspore import dtype as mstype
from mindspore import ops, Tensor
from mindspore_federated._mindspore_federated import TensorListItem_, TensorItem_


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
    data_type = "float32"
    if data_type is None:
        raise TypeError('tensor_to_tensor_pybind_obj: input a Tensor with unsupported value type ', ts.dtype)
    tensor.set_dtype(data_type)
    tensor.set_shape(ts.shape)
    ts_values = ts.reshape(ts.size,).asnumpy()
    tensor.set_data(ts_values)
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
    ts_type = mstype.float32
    if ts_type is None:
        logging.warning('tensor_pybind_obj_to_tensor: not specify data_type in TensorProto, using float32 by default')
        ts_type = mstype.float32
    values = tensor_item.data()
    ts = Tensor(list(values), dtype=ts_type)

    ts = ops.reshape(ts, tuple(tensor_item.shape()))
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
