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

from mindspore import Tensor
from mindspore_federated._mindspore_federated import TensorListItem_, TensorItem_
from mindspore_federated import log as logger
from ..startup.compress_config import COMPRESS_TYPE_FUNC_DICT, DECOMPRESS_TYPE_FUNC_DICT,\
    NO_COMPRESS_TYPE, COMPRESS_SUPPORT_NPTYPES, CompressConfig


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


def tensor_to_tensor_pybind_obj(ts, ref_key, compress_config=None):
    """
    Convert a tensor to a pybind item.
    Inputs:
        ts (Tensor): the Tensor object.
        ref_key (str): the ref name of the Tensor object.
        compress_config (CompressConfig): Configuration of compress communication. Default: None.
    """
    if ts is None:
        raise ValueError('tensor_to_tensor_pybind_obj: could not input a Tensor with None value')
    if ts.dtype is None:
        raise TypeError('tensor_to_tensor_pybind_obj: input a Tensor with unsupported value type.')
    np_data = ts.reshape(ts.size,).asnumpy()
    dtype_str = str(np_data.dtype)
    logger.info("Encode begin.")
    if compress_config is not None and dtype_str in COMPRESS_SUPPORT_NPTYPES:
        if compress_config.compress_type in COMPRESS_TYPE_FUNC_DICT:
            compress_func = COMPRESS_TYPE_FUNC_DICT[compress_config.compress_type]
            tensor = compress_func(list(np_data), compress_config.bit_num)
            compress_type = tensor.compress_type()
            logger.info("The compress_type is {}.".format(compress_type))
            if compress_type == NO_COMPRESS_TYPE:
                np_data_b64 = base64.b64encode(np_data.tobytes())
                tensor.set_raw_data(np_data_b64)
        else:
            raise ValueError('compress_type: {} is not supported now.'.format(compress_config.compress_type))
    else:
        tensor = TensorItem_()
        tensor.set_compress_type(NO_COMPRESS_TYPE)
        np_data_b64 = base64.b64encode(np_data.tobytes())
        tensor.set_raw_data(np_data_b64)
    logger.info("Encode end.")
    raw_data_size = tensor.raw_data_size()
    logger.info("The send size of {} is about {} B.".format(ref_key, raw_data_size))
    if ref_key:
        tensor.set_ref_key(ref_key)
    tensor.set_dtype(dtype_str)
    tensor.set_shape(ts.shape)

    return tensor


def tensor_dict_to_tensor_list_pybind_obj(ts_dict, name, compress_configs=None):
    """
    Convert a dict, the s of which are tensor objects, to pybind object.
    Inputs:
        ts_dict (dict): the dict object, the items of which are tensors.
        name (str): the ref name of the pybind object.
        compress_configs (dict): Configurations of compress communication. Default: None.
    """
    tensor_list_item = TensorListItem_()
    tensor_list_item.set_name(name)
    for ts_key, ts in ts_dict.items():
        if isinstance(ts, Tensor):
            if ts_key in compress_configs:
                compress_config = compress_configs[ts_key]
                if not isinstance(compress_config, CompressConfig):
                    raise ValueError(
                        "compress_config of {} is not a type of CompressConfig, but get {}".format(
                            ts_key, type(compress_config)))
            else:
                compress_config = None
            tensor = tensor_to_tensor_pybind_obj(ts, ts_key, compress_config)
            tensor_list_item.add_tensor(tensor)
        elif isinstance(ts, OrderedDict):
            sub_tensor_list_item = tensor_dict_to_tensor_list_pybind_obj(ts, ts_key, compress_configs)
            tensor_list_item.add_tensor_list_item(sub_tensor_list_item)
        else:
            raise ValueError('Tensor type is invalid, type must be Tensor or OrderedDict, but get {}'.format(type(ts)))
    return tensor_list_item


def tensor_pybind_obj_to_tensor(tensor_item):
    """
    Convert a pybind to a tensor.
    Inputs:
        tensor_item (TensorItem_): the pybind object of the Tensor.
    """
    ref_key = tensor_item.ref_key()
    dtype_str = tensor_item.dtype()
    compress_type = tensor_item.compress_type()
    shape = tensor_item.shape()
    if dtype_str is None or dtype_str == "":
        raise ValueError('tensor_pybind_obj_to_tensor: Tensor with unsupported value type.')
    dtype = DATATYPE_STRING_NPTYPE_DICT.get(dtype_str)
    logger.info("Decode begin.")
    if compress_type in DECOMPRESS_TYPE_FUNC_DICT:
        decompress_func = DECOMPRESS_TYPE_FUNC_DICT[compress_type]
        logger.info("The compress_type from remote is {}.".format(compress_type))
        list_data = decompress_func(tensor_item)
        np_data = np.array(list_data, dtype=dtype).reshape(shape)
    elif compress_type == NO_COMPRESS_TYPE:
        raw_data = tensor_item.raw_data()
        values = base64.b64decode(raw_data)
        np_data = np.frombuffer(values, dtype=dtype).reshape(shape)
    else:
        raise ValueError('compress_type: {} is not supported now.'.format(compress_type))
    logger.info("Decode end.")
    raw_data_size = tensor_item.raw_data_size()
    logger.info("The receive size of {} is about {} B.".format(ref_key, raw_data_size))
    ts = Tensor(np_data)
    return ref_key, ts


def tensor_list_pybind_obj_to_tensor_dict(tensor_list_item):
    """
    Parse a dict, the s of which are tensor objects, from a pybind object.
    Inputs:
        tensor_list_item (TensorListItem_): the pybind object.
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
