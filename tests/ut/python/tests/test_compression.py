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
"""Test the functions of communication compression"""
import os
from collections import OrderedDict
from multiprocessing import Process
import numpy as np
from mindspore import Tensor
from mindspore_federated.startup.vertical_federated_local import VerticalFederatedCommunicator, ServerConfig
from mindspore_federated.startup.compress_config import CompressConfig
from common import communication_compression_test


def _parse_deep_dict(deep_dict, outputs):
    """
    Parse the values of a dictionary.
    """
    for key in deep_dict:
        if isinstance(deep_dict[key], OrderedDict):
            _parse_deep_dict(deep_dict[key], outputs)
        else:
            outputs.append(deep_dict[key].asnumpy())


def leader_process_fun():
    """start vfl leader"""
    # get compress config
    compress_configs = {
        "embedding_table": CompressConfig("min_max", 6),
        "attention_mask": CompressConfig("bit_pack", 1),
        "word_table": CompressConfig("min_max", 7),
    }

    # build vfl communicator
    http_server_config = ServerConfig(server_name='leader', server_address="127.0.0.1:6969")
    remote_server_config = ServerConfig(server_name='follower', server_address="127.0.0.1:9696")
    vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
                                                          remote_server_config=remote_server_config,
                                                          compress_configs=compress_configs)
    vertical_communicator.launch()

    # fake data
    embedding_table = Tensor(np.random.random(size=(2, 1024, 1280)).astype(np.float32))
    attention_mask = Tensor(np.random.randint(0, 2, size=(2, 1024, 1024)).astype(np.float16))
    output = Tensor(np.random.random(size=(1,)).astype(np.float32))
    input_ids = Tensor(np.random.randint(0, 30000, size=(2, 1025)).astype(np.int32))
    position_id = Tensor(np.arange(start=0, stop=1024, step=1, dtype=np.int32))
    word_table = Tensor(np.random.random(size=(40000, 1280)).astype(np.float32))

    # fake message
    embedding_out = OrderedDict()
    embedding_out["embedding_table"] = embedding_table
    embedding_out["attention_mask"] = attention_mask
    head_scale = OrderedDict()
    head_scale["output"] = output
    head_scale["input_ids"] = input_ids
    head_scale["position_id"] = position_id
    head_scale["attention_mask"] = attention_mask
    head_scale["word_table"] = word_table

    # communication
    vertical_communicator.send_tensors("follower", embedding_out)
    backbone_out = vertical_communicator.receive("follower")
    vertical_communicator.send_tensors("follower", head_scale)
    backbone_scale = vertical_communicator.receive("follower")

    # save result
    send_outputs = list()
    recv_outputs = list()
    _parse_deep_dict(embedding_out, send_outputs)
    _parse_deep_dict(head_scale, send_outputs)
    _parse_deep_dict(backbone_out, recv_outputs)
    _parse_deep_dict(backbone_scale, recv_outputs)
    for i, output in enumerate(send_outputs):
        np.save("temp/leader/send/{}".format(i), output)
    for i, output in enumerate(recv_outputs):
        np.save("temp/leader/recv/{}".format(i), output)


def follower_process_fun():
    """start vfl follower"""
    # get compress config
    compress_configs = {
        "embedding_table": CompressConfig("min_max", 7),
        "hidden_states": CompressConfig("min_max", 8),
    }

    # build vfl communicator
    http_server_config = ServerConfig(server_name='follower', server_address="127.0.0.1:9696")
    remote_server_config = ServerConfig(server_name='leader', server_address="127.0.0.1:6969")
    vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
                                                          remote_server_config=remote_server_config,
                                                          compress_configs=compress_configs)
    vertical_communicator.launch()

    # fake data
    hidden_states = Tensor(np.random.random(size=(2, 2048, 1280)).astype(np.float16))
    embedding_table = Tensor(np.random.random(size=(2, 1024, 1280)).astype(np.float32))

    # fake message
    backbone_out = OrderedDict()
    backbone_out["hidden_states"] = hidden_states
    backbone_scale = OrderedDict()
    hidden_states_dict = OrderedDict()
    hidden_states_dict["embedding_table"] = embedding_table
    backbone_scale["hidden_states"] = hidden_states_dict

    # communication
    embedding_out = vertical_communicator.receive("leader")
    vertical_communicator.send_tensors("leader", backbone_out)
    head_scale = vertical_communicator.receive("leader")
    vertical_communicator.send_tensors("leader", backbone_scale)

    # save result
    send_outputs = list()
    recv_outputs = list()
    _parse_deep_dict(backbone_out, send_outputs)
    _parse_deep_dict(backbone_scale, send_outputs)
    _parse_deep_dict(embedding_out, recv_outputs)
    _parse_deep_dict(head_scale, recv_outputs)
    for i, output in enumerate(send_outputs):
        np.save("temp/follower/send/{}".format(i), output)
    for i, output in enumerate(recv_outputs):
        np.save("temp/follower/recv/{}".format(i), output)


@communication_compression_test
def test_vfl_communication_compression():
    """
    Feature: Test vfl communication compression: whole flow.
    Description: Input constructed through numpy.
    Expectation: ERROR log is right and success.
    """

    # communication
    leader_process = Process(target=leader_process_fun, args=())
    leader_process.start()
    follower_process = Process(target=follower_process_fun, args=())
    follower_process.start()

    # start and stop processes
    leader_process.join(timeout=200)
    follower_process.join(timeout=200)
    leader_process.terminate()
    follower_process.terminate()
    leader_process.kill()
    follower_process.kill()

    # load result
    leader_send_dir = "temp/leader/send"
    leader_recv_dir = "temp/leader/recv"
    follower_send_dir = "temp/follower/send"
    follower_recv_dir = "temp/follower/recv"
    leader_send_outputs = list()
    leader_recv_outputs = list()
    follower_send_outputs = list()
    follower_recv_outputs = list()
    for path in os.listdir(leader_send_dir):
        output = np.load(os.path.join(leader_send_dir, path))
        leader_send_outputs.append(output)
    for path in os.listdir(leader_recv_dir):
        output = np.load(os.path.join(leader_recv_dir, path))
        leader_recv_outputs.append(output)
    for path in os.listdir(follower_send_dir):
        output = np.load(os.path.join(follower_send_dir, path))
        follower_send_outputs.append(output)
    for path in os.listdir(follower_recv_dir):
        output = np.load(os.path.join(follower_recv_dir, path))
        follower_recv_outputs.append(output)

    # check results
    rtol = 1.e-2
    atol = 1.e-2
    assert len(leader_send_outputs) == len(follower_recv_outputs) == 7
    assert len(leader_recv_outputs) == len(follower_send_outputs) == 2
    for output1, output2 in zip(leader_send_outputs, follower_recv_outputs):
        assert np.allclose(output1, output2, rtol, atol)
    for output1, output2 in zip(leader_recv_outputs, follower_send_outputs):
        assert np.allclose(output1, output2, rtol, atol)
