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

import logging

from mindspore import context
from mindspore_federated import FLModel, FLYamlData
from mindspore_federated.startup.vertical_federated_local import VerticalFederatedCommunicator, ServerConfig
from mindspore_federated.data_join import FLDataWorker

from wide_and_deep import FollowerNet, FollowerLossNet
from network_config import config
from criteo_dataset import create_joined_dataset


class FollowerTrainer:
    """Process of follower party"""

    def __init__(self):
        super(FollowerTrainer, self).__init__()
        self.content = None
        http_server_config = ServerConfig(server_name='client', server_address=config.http_server_address)
        remote_server_config = ServerConfig(server_name='server', server_address=config.remote_server_address)
        self.vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
                                                                   remote_server_config=remote_server_config)
        self.vertical_communicator.launch()
        logging.info('start vfl trainer success')
        follower_yaml_data = FLYamlData(config.follower_yaml_path)
        follower_eval_net = follower_base_net = FollowerNet(config)
        follower_train_net = FollowerLossNet(follower_base_net, config)
        self.follower_fl_model = FLModel(yaml_data=follower_yaml_data,
                                         network=follower_train_net,
                                         eval_network=follower_eval_net)
        logging.info('Init follower trainer finish.')

    def start(self):
        """
        Run follower trainer
        """
        logging.info('Begin follower trainer')
        if config.resume:
            self.follower_fl_model.load_ckpt()
        for _ in range(config.epochs):
            for _, item in enumerate(train_iter):
                follower_out = self.follower_fl_model.forward_one_step(item)
                self.vertical_communicator.send_tensors("server", follower_out)
                scale = self.vertical_communicator.receive("server")
                self.follower_fl_model.backward_one_step(item, sens=scale)
            self.follower_fl_model.save_ckpt()
            for eval_item in eval_iter:
                follower_out = self.follower_fl_model.forward_one_step(eval_item)
                self.vertical_communicator.send_tensors("server", follower_out)


logging.basicConfig(filename='follower_process.log', level=logging.INFO)
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

logging.info("config is:")
logging.info(config)

worker = FLDataWorker(role="follower",
                      main_table_files=config.raw_dataset_dir,
                      output_dir=config.dataset_dir,
                      data_schema_path=config.data_schema_path,
                      http_server_address=config.http_server_address,
                      remote_server_address=config.remote_server_address,
                      primary_key=config.primary_key,
                      bucket_num=config.bucket_num,
                      store_type=config.store_type,
                      shard_num=config.shard_num,
                      join_type=config.join_type,
                      thread_num=config.thread_num,
                      )
worker.export()
logging.info('train dataset export is done')
ds_train = create_joined_dataset(config.dataset_dir, batch_size=config.batch_size, train_mode=True,
                                 role="follower")
train_iter = ds_train.create_dict_iterator()
train_size = ds_train.get_dataset_size()
logging.info("train_size is: %d", train_size)

worker = FLDataWorker(role="follower",
                      main_table_files=config.raw_eval_dataset_dir,
                      output_dir=config.eval_dataset_dir,
                      data_schema_path=config.data_schema_path,
                      http_server_address=config.http_server_address,
                      remote_server_address=config.remote_server_address,
                      primary_key=config.primary_key,
                      bucket_num=config.bucket_num,
                      store_type=config.store_type,
                      shard_num=config.shard_num,
                      join_type=config.join_type,
                      thread_num=config.thread_num,
                      )
worker.export()
logging.info('eval dataset export is done')
ds_eval = create_joined_dataset(config.eval_dataset_dir, batch_size=config.batch_size, train_mode=False,
                                role="follower")
eval_iter = ds_eval.create_dict_iterator()
eval_size = ds_eval.get_dataset_size()
logging.info("eval_size is: %d", eval_size)


if __name__ == '__main__':
    follower_trainer = FollowerTrainer()
    follower_trainer.start()
    logging.info("train is done")
