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

import os.path
import struct
from typing import OrderedDict

import logging
import yaml

from mindspore import nn, ops, Tensor, ParameterTuple
from mindspore import dtype as mstype

from .vfl_pb2 import DataType, TensorListProto, TensorProto



def parse_yaml_file(file_path):
    yaml_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        assert ValueError(f'File {yaml_path} not exit')
    with open(yaml_path, 'r', encoding='utf-8') as fp:
        yaml_data = yaml.safe_load(fp)
    return yaml_data, fp


class YamlDataParser:
    """
    Parse the yaml file describing the vfl process, and generate corresponding data structures.

    Args:
        yaml_path: path to the yaml file
    """
    def __init__(self, yaml_data):
        self.role = yaml_data['role']
        model_data = yaml_data['model']
        self.train_net = model_data['train_net']
        self.eval_net = model_data['eval_net']
        self.opt_list = yaml_data['opts']
        if 'grad_scales' in yaml_data:
            self.grad_scaler_list = yaml_data['grad_scales']
        self.dataset = yaml_data['dataset']
        self.train_hyper_parameters = yaml_data['train_hyper_parameters']


def get_params_list_by_name(net, name):
    """
    Get parameters list by name from the nn.Cell

    Inputs:
        net (nn.Cell): Network described using mindspore.
        name (str): Name of parameters to be gotten.
    """
    res = []
    trainable_params = net.trainable_params()
    for param in trainable_params:
        if name in param.name:
            res.append(param)
    return res


def get_params_by_name(net, weight_name_list):
    """
    Get parameters list by names from the nn.Cell

    Inputs:
        net (nn.Cell): Network described using mindspore.
        name (list): Names of parameters to be gotten.
    """
    params = []
    for weight_name in weight_name_list:
        params.extend(get_params_list_by_name(net, weight_name))
    params = ParameterTuple(params)
    return params


class IthOutputCellInDict(nn.Cell):
    """
    Encapulate network with multiple outputs so that it only output one variable.

    Args:
        network (nn.Cell): Network to be encapulated.
        output_index (int): Index of the output variable.

    Inputs:
        **kwargs (dict): input of the network.
    """
    def __init__(self, network, output_index):
        super(IthOutputCellInDict, self).__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, **kwargs):
        return self.network(**kwargs)[self.output_index]


class IthOutputCellInTuple(nn.Cell):
    """
    Encapulate network with multiple outputs so that it only output one variable.

    Args:
        network (nn.Cell): Network to be encapulated.
        output_index (int): Index of the output variable.

    Inputs:
        *kwargs (tuple): input of the network.
    """
    def __init__(self, network, output_index):
        super(IthOutputCellInTuple, self).__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, *args):
        return self.network(*args)[self.output_index]


MSTYPE_DATATYPE_DICT = {
    mstype.float32: DataType.FLOAT,
    mstype.uint8: DataType.UINT8,
    mstype.int8: DataType.INT8,
    mstype.uint16: DataType.UINT16,
    mstype.int16: DataType.INT16,
    mstype.int32: DataType.INT32,
    mstype.int64: DataType.INT64,
    mstype.string: DataType.STRING,
    mstype.bool_: DataType.BOOL,
    mstype.float16: DataType.FLOAT16,
    mstype.double: DataType.DOUBLE,
    mstype.uint32: DataType.UINT32,
    mstype.uint64: DataType.UINT64,
    mstype.float64: DataType.FLOAT64
}

DATATYPE_MSTYPE_DICT = {
    DataType.FLOAT: mstype.float32,
    DataType.UINT8: mstype.uint8,
    DataType.INT8: mstype.int8,
    DataType.UINT16: mstype.uint16,
    DataType.INT16: mstype.int16,
    DataType.INT32: mstype.int32,
    DataType.INT64: mstype.int64,
    DataType.STRING: mstype.string,
    DataType.BOOL: mstype.bool_,
    DataType.FLOAT16: mstype.float16,
    DataType.DOUBLE: mstype.double,
    DataType.UINT32: mstype.uint32,
    DataType.UINT64: mstype.uint64,
    DataType.FLOAT64: mstype.float64
}

PROTO_DATA_MAP = {
    DataType.BOOL: 'int32_data',
    DataType.STRING: 'string_data',
    DataType.INT8: 'int32_data',
    DataType.INT16: 'int32_data',
    DataType.INT32: 'int32_data',
    DataType.INT64: 'int64_data',
    DataType.UINT8: 'uint32_data',
    DataType.UINT16: 'uint32_data',
    DataType.UINT32: 'uint32_data',
    DataType.UINT64: 'uint64_data',
    DataType.FLOAT: 'float_data',
    DataType.FLOAT16: 'float_data',
    DataType.DOUBLE: 'double_data',
    DataType.FLOAT64: 'double_data'
}


def get_proto_data_by_type(ts_proto: TensorProto, data_type):
    """
    Get the data type of a Tensor to construct a proto object.

    Inputs:
        ts_proto (class): the proto object of the tensor.
        data_type (enum): the data type of the tensor.
    """
    if not ts_proto:
        raise ValueError('get_proto_data_by_type: could not input a None ts_proto')
    if data_type not in PROTO_DATA_MAP:
        raise TypeError('get_proto_data_by_type: unsupported data_type ', data_type)
    data_attr_name = PROTO_DATA_MAP[data_type]
    return getattr(ts_proto, data_attr_name)


def tensor_to_proto(ts: Tensor, ref_key: str = None):
    """
    Convert a tensor to a proto.
    Inputs:
        ts (class): the Tensor object.
        ref_key (str): the ref name of the Tensor object.
    """
    if ts is None:
        raise ValueError('tensor_to_proto: could not input a Tensor with None value')
    ts_proto = TensorProto()
    if ref_key:
        ts_proto.ref_key = ref_key
    data_type = MSTYPE_DATATYPE_DICT.get(ts.dtype)
    if data_type is None:
        raise TypeError('tensor_to_proto: input a Tensor with unsupported value type ', ts.dtype)
    ts_proto.data_type = data_type
    ts_proto.dims.extend(ts.shape)
    ts_values = ts.reshape(ts.size,).asnumpy()
    get_proto_data_by_type(ts_proto, data_type).extend(ts_values)
    return ts_proto


def proto_to_tensor(ts_proto: TensorProto):
    """
    Convert a proto to a tensor.
    Inputs:
        ts_proto (class): the proto object of the Tensor.
    """
    ref_key = ts_proto.ref_key
    ts_type = DATATYPE_MSTYPE_DICT.get(ts_proto.data_type)
    if ts_type is None:
        logging.warning('proto_to_tensor: not specify data_type in TensorProto, using float32 by default')
        ts_type = mstype.float32
    values = get_proto_data_by_type(ts_proto, ts_proto.data_type)
    ts = Tensor(list(values), dtype=ts_type)
    ts = ops.reshape(ts, tuple(ts_proto.dims))
    return ref_key, ts


def tensor_dict_to_proto(ts_dict: dict, name: str = 'no_name', send_addr: str = 'localhost',
                         recv_addr: str = 'localhost'):
    """
    Convert a dict, the s of which are tensor objects, to proto object.
    Inputs:
        ts_dict (dict): the dict object, the items of which are tensors.
        name (str): the ref name of the proto object.
        send_addr (str): the send address of the proto object.
        recv_addr (str): the receive address of the proto object.
    """
    ts_list_proto = TensorListProto()
    ts_list_proto.name = name
    ts_list_proto.send_addr = send_addr
    ts_list_proto.recv_addr = recv_addr
    ts_list_proto.length = len(ts_dict)
    for ts_key, ts in ts_dict.items():
        if isinstance(ts, Tensor):
            ts_list_proto.tensors.append(tensor_to_proto(ts, ts_key))
        elif isinstance(ts, OrderedDict):
            ts_list_proto.tensor_list.append(tensor_dict_to_proto(ts, ts_key, send_addr, recv_addr))
    return ts_list_proto


def proto_to_tensor_dict(ts_list_proto: TensorListProto):
    """
    Parse a dict, the s of which are tensor objects, from a proto object.
    Inputs:
        ts_list_proto (class): the proto object.
    """
    name = ts_list_proto.name
    res = OrderedDict()
    tensors = list(ts_list_proto.tensors)
    for ts_proto in tensors:
        ref_key, ts = proto_to_tensor(ts_proto)
        res[ref_key] = ts
    tensor_dicts = list(ts_list_proto.tensor_list)
    for ts_list_proto_item in tensor_dicts:
        sub_name, sub_res = proto_to_tensor_dict(ts_list_proto_item)
        res[sub_name] = sub_res
    return name, res


def socket_read_nbytes(sock, n):
    """
    read n bytes from socket connection
    """
    buf = b''
    while n > 0:
        data = sock.recv(n)
        if not data:
            raise SystemError('unexpected socket connection close')
        buf += data
        n -= len(data)
    return buf


def send_proto(sock, proto_msg):
    """
    send a proto message through the socket connection.
    """
    msg = proto_msg.SerializeToString()
    msg_len = struct.pack('>L', len(msg))
    sock.sendall(msg_len + msg)
    return len(msg)


def recv_proto(sock):
    """
    receive a proto message from the socket connection.
    """
    msg_len = socket_read_nbytes(sock, 4)
    msg_len = struct.unpack('>L', msg_len)[0]
    msg_buf = socket_read_nbytes(sock, msg_len)
    return msg_buf, msg_len
