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
Execute Wide&Deep split nn demo leader training on Criteo dataset with type of MindRecord.
The embeddings and grad scales are transmitted through http.
"""

import logging
from collections import OrderedDict

from mindspore import set_seed
from mindspore import context
from mindspore_federated import FLModel, FLYamlData
from mindspore_federated.privacy import LabelDP
from mindspore_federated.startup.vertical_federated_local import VerticalFederatedCommunicator, ServerConfig
from wide_and_deep import WideDeepModel, BottomLossNet, LeaderTopNet, LeaderTopLossNet, LeaderTopEvalNet, \
     LeaderTeeNet, LeaderTeeLossNet, LeaderTopAfterTeeNet, LeaderTopAfterTeeLossNet, LeaderTopAfterTeeEvalNet, \
     AUCMetric

from network_config import config
from run_vfl_train_local import construct_local_dataset


set_seed(0)


class LeaderTrainer:
    """Process of leader party"""

    def __init__(self):
        super(LeaderTrainer, self).__init__()
        logging.info('start vfl trainer success')

        if config.simu_tee:
            # parse yaml files
            leader_bottom_yaml_data = FLYamlData(config.leader_bottom_tee_yaml_path)
            leader_tee_yaml_data = FLYamlData(config.leader_tee_yaml_path)
            leader_top_yaml_data = FLYamlData(config.leader_top_tee_yaml_path)
            # Leader Tee Net
            leader_tee_eval_net = leader_tee_base_net = LeaderTeeNet()
            leader_tee_train_net = LeaderTeeLossNet(leader_tee_base_net)
            self.leader_tee_fl_model = FLModel(yaml_data=leader_tee_yaml_data,
                                               network=leader_tee_train_net,
                                               eval_network=leader_tee_eval_net)
            # Leader Top Net
            leader_top_base_net = LeaderTopAfterTeeNet()
            leader_top_train_net = LeaderTopAfterTeeLossNet(leader_top_base_net)
            leader_top_eval_net = LeaderTopAfterTeeEvalNet(leader_top_base_net)
        else:
            # parse yaml files
            leader_bottom_yaml_data = FLYamlData(config.leader_bottom_yaml_path)
            leader_top_yaml_data = FLYamlData(config.leader_top_yaml_path)
            # Leader Top Net
            leader_top_base_net = LeaderTopNet()
            leader_top_train_net = LeaderTopLossNet(leader_top_base_net)
            leader_top_eval_net = LeaderTopEvalNet(leader_top_base_net)

        self.eval_metric = AUCMetric()
        self.leader_top_fl_model = FLModel(yaml_data=leader_top_yaml_data,
                                           network=leader_top_train_net,
                                           metrics=self.eval_metric,
                                           eval_network=leader_top_eval_net)

        self.ldp = None
        if hasattr(leader_top_yaml_data, 'label_dp_eps') and config.label_dp:
            self.ldp = LabelDP(leader_top_yaml_data.label_dp_eps)

        # Leader Bottom Net
        leader_bottom_eval_net = leader_bottom_base_net = WideDeepModel(config, config.leader_field_size)
        leader_bottom_train_net = BottomLossNet(leader_bottom_base_net, config)
        self.leader_bottom_fl_model = FLModel(yaml_data=leader_bottom_yaml_data,
                                              network=leader_bottom_train_net,
                                              eval_network=leader_bottom_eval_net)

        # get compress config
        compress_configs = self.leader_top_fl_model.get_compress_configs()

        # build vertical communicator
        http_server_config = ServerConfig(server_name='leader', server_address=config.http_server_address)
        remote_server_config = ServerConfig(server_name='follower', server_address=config.remote_server_address)
        self.vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
                                                                   remote_server_config=remote_server_config,
                                                                   compress_configs=compress_configs)
        self.vertical_communicator.launch()
        logging.info('Init leader trainer finish.')

    def start(self):
        """
        Run leader trainer
        """
        logging.info('Begin leader trainer')
        if config.resume:
            self.leader_top_fl_model.load_ckpt()
            self.leader_bottom_fl_model.load_ckpt()
            if config.simu_tee:
                self.leader_tee_fl_model.load_ckpt()
        for epoch in range(config.epochs):
            for step, item in enumerate(train_iter):
                if self.ldp:
                    item['label'] = self.ldp(item['label'])

                leader_embedding = self.leader_bottom_fl_model.forward_one_step(item)
                item.update(leader_embedding)
                follower_embedding = self.vertical_communicator.receive("follower")

                if config.simu_tee:
                    tee_embedding = self.leader_tee_fl_model.forward_one_step(item, follower_embedding)
                    leader_out = self.leader_top_fl_model.forward_one_step(item, tee_embedding)
                    grad_scale = self.leader_top_fl_model.backward_one_step(item, tee_embedding)
                    grad_scale = self.leader_tee_fl_model.backward_one_step(item, follower_embedding, sens=grad_scale)
                    self.leader_tee_fl_model.save_ckpt()
                    scale_name = 'tee_embedding'
                else:
                    leader_out = self.leader_top_fl_model.forward_one_step(item, follower_embedding)
                    grad_scale = self.leader_top_fl_model.backward_one_step(item, follower_embedding)
                    scale_name = 'loss'

                grad_scale_follower = {scale_name: OrderedDict(list(grad_scale[scale_name].items())[2:])}
                self.vertical_communicator.send_tensors("follower", grad_scale_follower)
                grad_scale_leader = {scale_name: OrderedDict(list(grad_scale[scale_name].items())[:2])}
                self.leader_bottom_fl_model.backward_one_step(item, sens=grad_scale_leader)

                if step % 100 == 0:
                    logging.info('epoch %d step %d loss: %f', epoch, step, leader_out['loss'])

            self.leader_bottom_fl_model.save_ckpt()
            self.leader_top_fl_model.save_ckpt()

            for eval_item in eval_iter:
                leader_embedding = self.leader_bottom_fl_model.forward_one_step(eval_item)
                follower_embedding = self.vertical_communicator.receive("follower")
                embedding = follower_embedding
                embedding.update(leader_embedding)

                if config.simu_tee:
                    tee_embedding = self.leader_tee_fl_model.forward_one_step(eval_item, embedding)
                    _ = self.leader_top_fl_model.eval_one_step(eval_item, tee_embedding)
                else:
                    _ = self.leader_top_fl_model.eval_one_step(eval_item, embedding)

            auc = self.eval_metric.eval()
            self.eval_metric.clear()
            logging.info('epoch %d auc: %f', epoch, auc)


logging.basicConfig(filename='leader_train.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
ds_train, ds_eval = construct_local_dataset()
train_iter = ds_train.create_dict_iterator()
eval_iter = ds_eval.create_dict_iterator()
train_size = ds_train.get_dataset_size()
eval_size = ds_eval.get_dataset_size()
logging.info("train_size is: %d", train_size)
logging.info("eval_size is: %d", eval_size)


if __name__ == '__main__':
    leader_trainer = LeaderTrainer()
    leader_trainer.start()
