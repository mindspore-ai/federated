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
"""
Local splitnn of wide and deep on criteo dataset.
The embeddings and grad-scales are transmitted through socket.
"""

import socket
import logging
import threading
import itertools

from mindspore import context

from mindspore_federated import FLModel, FLYamlData, vfl_utils

from run_vfl_train_local import construct_local_dataset
from network_config import config
from wide_and_deep import LeaderNet, LeaderLossNet, LeaderEvalNet, \
    FollowerNet, FollowerLossNet, AUCMetric


class LeaderThread(threading.Thread):
    """Thread of leader party"""
    def __init__(self):
        super(LeaderThread, self).__init__()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('0.0.0.0', 6001))
        self.socket.listen(5)

        leader_yaml_data = FLYamlData(config.leader_yaml_path)
        leader_base_net = LeaderNet(config)
        leader_train_net = LeaderLossNet(leader_base_net, config)
        leader_eval_net = LeaderEvalNet(leader_base_net)
        self.eval_metric = AUCMetric()
        self.leader_fl_model = FLModel(role='leader',
                                       yaml_data=leader_yaml_data,
                                       network=leader_base_net,
                                       train_network=leader_train_net,
                                       metrics=self.eval_metric,
                                       eval_network=leader_eval_net)
        self.embedding_proto = vfl_utils.TensorListProto()
        self.grad_scale_proto = vfl_utils.TensorListProto()

    def run(self) -> None:
        global current_item, train_size
        while True:
            client_socket, client_addr = self.socket.accept()
            logging.info('Connect with: %s', str(client_addr))
            for epoch, step in itertools.product(range(config.epochs), range(train_size)):
                msg, msg_len = vfl_utils.recv_proto(client_socket)
                logging.debug('leader recv bytes: %d', msg_len)
                self.embedding_proto.ParseFromString(msg)
                _, follower_out = vfl_utils.proto_to_tensor_dict(self.embedding_proto)
                leader_out = self.leader_fl_model.forward_one_step(current_item, follower_out)
                scale = self.leader_fl_model.backward_one_step(current_item, follower_out)
                self.grad_scale_proto = vfl_utils.tensor_dict_to_proto(scale)
                msg_len = vfl_utils.send_proto(client_socket, self.grad_scale_proto)
                logging.debug('leader send bytes: %d', msg_len)
                if getattr(client_socket, '_closed'):
                    logging.info('Client disconnect: epoch %d, step %d', epoch, step)
                if step % 100 == 0:
                    logging.info('epoch %d step %d/%d wide_loss: %f deep_loss: %f',
                                 epoch, step, train_size, leader_out['wide_loss'], leader_out['deep_loss'])


class FollowerThread(threading.Thread):
    """Thread of follower party"""
    def __init__(self):
        super(FollowerThread, self).__init__()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('127.0.0.1', 6001))

        follower_yaml_data = FLYamlData(config.follower_yaml_path)
        follower_eval_net = follower_base_net = FollowerNet(config)
        follower_train_net = FollowerLossNet(follower_base_net, config)
        self.follower_fl_model = FLModel(role='follower',
                                         yaml_data=follower_yaml_data,
                                         network=follower_base_net,
                                         train_network=follower_train_net,
                                         eval_network=follower_eval_net)
        self.embedding_proto = vfl_utils.TensorListProto()
        self.grad_scale_proto = vfl_utils.TensorListProto()

    def run(self) -> None:
        global current_item, train_iter
        for _, item in itertools.product(range(config.epochs), train_iter):
            current_item = item
            follower_out = self.follower_fl_model.forward_one_step(item)
            self.embedding_proto = vfl_utils.tensor_dict_to_proto(follower_out)
            msg_len = vfl_utils.send_proto(self.socket, self.embedding_proto)
            logging.debug('follower send bytes: %d', msg_len)
            msg, msg_len = vfl_utils.recv_proto(self.socket)
            logging.debug('follower recv bytes: %d', msg_len)
            self.grad_scale_proto.ParseFromString(msg)
            _, scale = vfl_utils.proto_to_tensor_dict(self.grad_scale_proto)
            self.follower_fl_model.backward_one_step(item, sens=scale)


logging.basicConfig(filename='log_mult_thread_socket.txt', level=logging.INFO)
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
ds_train, ds_eval = construct_local_dataset()
# Global variable transmitting joined data, which will be replaced with data joining module
train_iter = ds_train.create_dict_iterator()
eval_iter = ds_eval.create_dict_iterator()
train_size = ds_train.get_dataset_size()
eval_size = ds_eval.get_dataset_size()
current_item = None

if __name__ == '__main__':
    leader_thread = LeaderThread()
    follower_thread = FollowerThread()
    leader_thread.start()
    follower_thread.start()
    leader_thread.join()
    follower_thread.join()
