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
from mindspore_federated.startup.compress_config import CompressConfig
from mindspore_federated.compress import encode_executor, decode_executor
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

DATATYPE_STRING_QUANT_DICT = {
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64,
}
kNoCompressType = "no_compress"
kQuant = "quant"
kQuant8Bits = "quant_8_bits"


def get_compress_type(compress_config: CompressConfig):
    if compress_config.type == kQuant:
        return kQuant + "_" + str(compress_config.quant_bits) + "_bits"


def construct_compress_tensor(data_list: list, compress_config: CompressConfig, tensor: TensorItem_):
    if data_list is None:
        raise RuntimeError(
            f"Parameter 'data_list' should be list, but got {type(data_list)}")
    quant_bits = compress_config.quant_bits
    if compress_config.type == kQuant and 0 < quant_bits <= 255:
        min_val, max_val, compress_data_list = encode_executor.quant_compress(data_list, quant_bits)
        tensor.set_min_val(min_val)
        tensor.set_max_val(max_val)
        data_type = DATATYPE_STRING_QUANT_DICT.get(quant_bits)
    else:
        raise RuntimeError("Compress config is invalid.")
    return np.array(compress_data_list, dtype=data_type)


def decompress_tensor(raw_data, compress_type: str, tensor_item: TensorItem_):
    if raw_data is None:
        raise RuntimeError(
            f"Parameter 'raw_data' should be str, but got {type(raw_data)}")
    if compress_type == kQuant8Bits:
        quant_bits = 8
        data_type = DATATYPE_STRING_QUANT_DICT.get(quant_bits)
        np_data = np.frombuffer(raw_data, dtype=data_type).reshape(-1)
        min_val = tensor_item.min_val()
        max_val = tensor_item.max_val()
        decompress_data = decode_executor.quant_decompress(np_data, quant_bits, min_val, max_val)
    else:
        raise RuntimeError("Compress type is invalid from remote server.")
    return np.array(decompress_data, dtype=tensor_item.dtype())


def tensor_to_tensor_pybind_obj(ts: Tensor, ref_key: str, compress_config: CompressConfig):
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
    np_data = ts.reshape(ts.size, ).asnumpy()
    tensor.set_dtype(str(np_data.dtype))
    tensor.set_shape(ts.shape)
    if compress_config is not None:
        np_data = construct_compress_tensor(np_data, compress_config, tensor)
        compress_type = get_compress_type(compress_config)
        tensor.set_compress_type(compress_type)
    else:
        tensor.set_compress_type(kNoCompressType)
    np_data_b64 = base64.b64encode(np_data.tobytes())
    tensor.set_raw_data(np_data_b64)
    return tensor


def tensor_dict_to_tensor_list_pybind_obj(ts_dict: dict, name: str, compress_config: CompressConfig):
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
            tensor = tensor_to_tensor_pybind_obj(ts, ts_key, compress_config)
            tensor_list_item.add_tensor(tensor)
        elif isinstance(ts, OrderedDict):
            sub_tensor_list_item = tensor_dict_to_tensor_list_pybind_obj(ts, ts_key, compress_config)
            tensor_list_item.add_tensor_list_item(sub_tensor_list_item)
        else:
            raise ValueError('Tensor type is invalid, type must be Tensor or OrderedDict, but get {}'.format(type(ts)))
    return tensor_list_item


def tensor_pybind_obj_to_tensor(tensor_item: TensorItem_):
    """
    Convert a pybind to a tensor.
    Inputs:
        tensor_item (class): the pybind object of the Tensor.
    """
    ref_key = tensor_item.ref_key()
    dtype_str = tensor_item.dtype()
    compress_type = tensor_item.compress_type()
    raw_data = tensor_item.raw_data()
    shape = tensor_item.shape()
    if dtype_str is None or dtype_str == "":
        raise ValueError('tensor_pybind_obj_to_tensor: Tensor with unsupported value type.')
    dtype = DATATYPE_STRING_NPTYPE_DICT.get(dtype_str)

    values = base64.b64decode(raw_data)
    if compress_type != kNoCompressType:
        np_data = decompress_tensor(values, compress_type, tensor_item)
    else:
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
