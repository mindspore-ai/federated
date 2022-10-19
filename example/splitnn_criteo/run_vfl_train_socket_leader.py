# pylint: pointless-string-statement
# pylint: missing-docstring

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
from mindspore_federated import FLModel, FLYamlData
from mindspore_federated import VerticalFederatedCommunicator, ServerConfig
from wide_and_deep import LeaderNet, LeaderLossNet, LeaderEvalNet, AUCMetric

from network_config import config
from run_vfl_train_local import construct_local_dataset


class LeaderTrainer:
    """Process of leader party"""

    def __init__(self):
        super(LeaderTrainer, self).__init__()
        self.data = None
        http_server_config = ServerConfig(server_name='serverB', server_address='10.113.216.44:6667')
        remote_server_config = ServerConfig(server_name='serverA', server_address='10.113.216.44:6666')
        self.vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
                                                                   remote_server_config=remote_server_config)
        self.vertical_communicator.launch()

        leader_yaml_data = FLYamlData(config.leader_yaml_path)
        leader_base_net = LeaderNet(config)
        leader_train_net = LeaderLossNet(leader_base_net, config)
        leader_eval_net = LeaderEvalNet(leader_base_net)
        self.eval_metric = AUCMetric()
        self.leader_fl_model = FLModel(yaml_data=leader_yaml_data,
                                       network=leader_train_net,
                                       metrics=self.eval_metric,
                                       eval_network=leader_eval_net)

    def Start(self):
        """
        Run leader trainer
        """
        while True:
            logging.info('Begin leader trainer')
            for _, item in itertools.product(range(config.epochs), train_iter):
                current_item = item
                follower_out = self.vertical_communicator.receive("serverA")
                leader_out = self.leader_fl_model.forward_one_step(current_item, follower_out)
                scale = self.leader_fl_model.backward_one_step(current_item, follower_out)
                self.vertical_communicator.send_tensors("serverA", scale)
                # if step % 10 == 0:
                logging.info('epoch %d step %d/%d wide_loss: %f deep_loss: %f',
                             1, 1, train_size, leader_out['wide_loss'], leader_out['deep_loss'])


logging.basicConfig(filename='log_mult_leader_process_socket.txt', level=logging.INFO)
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
ds_train, ds_eval = construct_local_dataset()
# Global variable transmitting joined data, which will be replaced with data joining module
train_iter = ds_train.create_dict_iterator()
eval_iter = ds_eval.create_dict_iterator()
train_size = ds_train.get_dataset_size()
eval_size = ds_eval.get_dataset_size()

if __name__ == '__main__':
    leader_trainer = LeaderTrainer()
    leader_trainer.Start()
