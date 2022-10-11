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

import itertools
import logging

from mindspore import context
from mindspore_federated import FLModel, vfl_utils, tensor_utils
from mindspore_federated.startup.vertical_federated_local import VerticalFederatedCommunicator, ServerConfig
from network.wide_and_deep import FollowerNet, FollowerLossNet

from network_config import config
from run_vfl_train_local import construct_local_dataset


class FollowerTrainer:
    """Process of follower party"""

    def __init__(self):
        super(FollowerTrainer, self).__init__()
        self.content = None
        http_server_config = ServerConfig(server_name='serverA', server_address='10.113.216.44:6666')
        remote_server_config = ServerConfig(server_name='serverB', server_address='10.113.216.44:6667')
        self.vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
                                                                   remote_server_config=remote_server_config)
        self.vertical_communicator.launch()
        logging.info('start vfl trainer success')
        follower_yaml_data, follower_fp = vfl_utils.parse_yaml_file(config.follower_yaml_path)
        follower_fp.close()
        follower_eval_net = follower_base_net = FollowerNet(config)
        follower_train_net = FollowerLossNet(follower_base_net, config)
        self.follower_fl_model = FLModel(role='follower',
                                         network=follower_base_net,
                                         train_network=follower_train_net,
                                         eval_network=follower_eval_net,
                                         yaml_data=follower_yaml_data)
        logging.info('Init follower trainer finish.')

    def Start(self):
        global current_item, train_iter
        for _, item in itertools.product(range(config.epochs), train_iter):
            current_item = item
            follower_out = self.follower_fl_model.forward_one_step(item)
            embedding_data = tensor_utils.tensor_dict_to_tensor_list_pybind_obj(follower_out)
            self.vertical_communicator.send("serverB", embedding_data)
            receive_msg = self.vertical_communicator.receive("serverB")
            _, scale = tensor_utils.tensor_list_pybind_obj_to_tensor_dict(receive_msg)
            self.follower_fl_model.backward_one_step(item, sens=scale)


logging.basicConfig(filename='log_mult_follower_process_socket.txt', level=logging.INFO)
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
ds_train, ds_eval = construct_local_dataset()
# Global variable transmitting joined data, which will be replaced with data joining module
train_iter = ds_train.create_dict_iterator()
eval_iter = ds_eval.create_dict_iterator()
train_size = ds_train.get_dataset_size()
eval_size = ds_eval.get_dataset_size()
current_item = None

if __name__ == '__main__':
    follower_trainer = FollowerTrainer()
    follower_trainer.Start()
