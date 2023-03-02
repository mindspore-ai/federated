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
"""Local splitnn of pangu_alpha on wiki dataset."""
import os
import logging

from mindspore import context
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.train.summary import SummaryRecord
from mindspore_federated import FLModel, FLYamlData
from mindspore_federated.privacy import EmbeddingDP
from mindspore_federated.startup.vertical_federated_local import VerticalFederatedCommunicator, ServerConfig

from src.split_pangu_alpha import PanGuHead, HeadLossNet, EmbeddingLayer, EmbeddingNecessaryLossNet, PPLMetric
from src.utils import LearningRate, get_args, construct_local_dataset, load_train_net, set_weight_decay, \
    set_embedding_weight_decay
from src.pangu_optim import PanguAlphaAdam, FP32StateAdamWeightDecay
from src.pangu_alpha_config import set_parse

project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")


class LeaderTrainer:
    """Process of leader party"""

    def __init__(self):
        super(LeaderTrainer).__init__()

        # read, parse and check the .yaml files of sub-networks
        embedding_yaml = FLYamlData('./embedding_https.yaml')
        self.edp = None
        if hasattr(embedding_yaml, 'embedding_dp_eps') and opt.embedding_dp:
            self.edp = EmbeddingDP(embedding_yaml.embedding_dp_eps)
        head_yaml = FLYamlData('./head_https.yaml')

        # local data iteration for experiment
        ds_train = construct_local_dataset(opt, rank_id, device_num)
        self.train_iter = ds_train.create_dict_iterator()
        self.train_size = ds_train.get_dataset_size()
        # eval data iteration for example
        ds_eval = construct_local_dataset(opt, rank_id, device_num, is_training=False)
        self.eval_iter = ds_eval.create_dict_iterator()
        self.eval_size = ds_eval.get_dataset_size()

        # loss scale
        lr = LearningRate(learning_rate=opt.start_lr, end_learning_rate=opt.end_lr,
                          warmup_steps=opt.warmup_step, decay_steps=200000)

        loss, config = load_train_net(opt)
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=1024.0, scale_factor=2, scale_window=1000)
        self.eval_metric = PPLMetric(opt.seq_length)

        # Embedding/Tail Part
        embedding_base_net = EmbeddingLayer(config)
        embedding_eval_net = embedding_train_net = EmbeddingNecessaryLossNet(embedding_base_net, config)
        embedding_with_loss = _VirtualDatasetCell(embedding_eval_net)
        embedding_params = embedding_with_loss.trainable_params()
        embedding_group_params = set_embedding_weight_decay(embedding_params)
        embedding_optim_inst = FP32StateAdamWeightDecay(embedding_group_params, lr, eps=1e-8, beta1=0.9, beta2=0.95)
        embedding_optim = PanguAlphaAdam(embedding_train_net, embedding_optim_inst, update_cell, config, embedding_yaml)

        # Head Party
        head_base_net = PanGuHead(config)
        head_eval_net = head_train_net = HeadLossNet(head_base_net, loss, config)
        head_with_loss = _VirtualDatasetCell(head_train_net)
        head_params = head_with_loss.trainable_params()
        head_group_params = set_weight_decay(head_params)
        head_optim_inst = FP32StateAdamWeightDecay(head_group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95)
        head_optim = PanguAlphaAdam(head_train_net, head_optim_inst, update_cell, config, head_yaml)

        # FLModel definition
        self.head_fl_model = FLModel(yaml_data=head_yaml,
                                     network=head_train_net,
                                     eval_network=head_eval_net,
                                     optimizers=head_optim,
                                     metrics=self.eval_metric)

        self.embedding_fl_model = FLModel(yaml_data=embedding_yaml,
                                          network=embedding_train_net,
                                          eval_network=embedding_eval_net,
                                          optimizers=embedding_optim)
        # load ckpt
        if opt.resume:
            self.embedding_fl_model.load_ckpt()
            self.head_fl_model.load_ckpt()

        # get compress config
        embedding_fl_compress_configs = self.embedding_fl_model.get_compress_configs()
        head_fl_compress_configs = self.head_fl_model.get_compress_configs()
        compress_configs = {**embedding_fl_compress_configs, **head_fl_compress_configs}

        # build vertical communicator
        http_server_config = ServerConfig(server_name='leader', server_address=opt.http_server_address)
        remote_server_config = ServerConfig(server_name='follower', server_address=opt.remote_server_address)
        self.vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
                                                                   remote_server_config=remote_server_config,
                                                                   compress_configs=compress_configs)
        self.vertical_communicator.launch()

    def start(self):
        """
        Run leader trainer
        """
        # forward/backward batch by batch
        with SummaryRecord('./summary') as summary_record:
            for epoch in range(50):
                for step, item in enumerate(self.train_iter, start=1):
                    step = epoch * self.train_size + step
                    embedding_out = self.embedding_fl_model.forward_one_step(item)
                    if self.edp:
                        embedding_out['embedding_table'] = self.edp(embedding_out['embedding_table'])
                    word_table_ts = embedding_out.pop('word_table')
                    self.vertical_communicator.send_tensors("follower", embedding_out)
                    backbone_out = self.vertical_communicator.receive("follower")
                    backbone_out['word_table'] = word_table_ts
                    logit_out = self.head_fl_model.forward_one_step(item, backbone_out)

                    # backward process
                    head_scale = self.head_fl_model.backward_one_step(item, backbone_out)
                    word_table_scale_ts = head_scale['output'].pop('word_table')
                    self.vertical_communicator.send_tensors("follower", head_scale)
                    backbone_scale = self.vertical_communicator.receive("follower")
                    head_scale['output']['word_table'] = word_table_scale_ts
                    backbone_scale.update(head_scale)
                    self.embedding_fl_model.backward_one_step(item, sens=backbone_scale)
                    if step % 10 == 0:
                        summary_record.add_value('scalar', 'output', logit_out['output'])
                        summary_record.record(step)
                        logging.info('epoch %d step %d/%d loss: %f', epoch, step - epoch*self.train_size,
                                     self.train_size, logit_out['output'])

                    if step % 1000 == 0:
                        # save checkpoint
                        self.embedding_fl_model.save_ckpt()
                        self.head_fl_model.save_ckpt()


if __name__ == '__main__':
    device_num = 1
    rank_id = 0
    opt = get_args()
    set_parse(opt)
    logging.basicConfig(filename='leader_process.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("config is:")
    logging.info(opt)
    context.set_context(mode=context.GRAPH_MODE, device_target=opt.device_target, device_id=opt.device_id)

    leader_trainer = LeaderTrainer()
    leader_trainer.start()
