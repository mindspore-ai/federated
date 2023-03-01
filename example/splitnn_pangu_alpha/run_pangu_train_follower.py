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
"""Splitnn of pangu_alpha on wiki dataset."""
import os
import logging

from mindspore import context
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore_federated import FLModel, FLYamlData
from mindspore_federated.startup.vertical_federated_local import VerticalFederatedCommunicator, ServerConfig

from src.split_pangu_alpha import PanguAlphaModel, BackboneNecessrayLossNet

from src.utils import LearningRate, get_args, load_train_net, set_weight_decay
from src.pangu_optim import PanguAlphaAdam, FP32StateAdamWeightDecay
from src.pangu_alpha_config import set_parse

project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")


class FollowerTrainer:
    """Process of follower party"""

    def __init__(self):
        super(FollowerTrainer).__init__()

        # read, parse and check the .yaml files of sub-networks
        backbone_yaml = FLYamlData('./backbone_https.yaml')

        # loss scale
        lr = LearningRate(learning_rate=opt.start_lr, end_learning_rate=opt.end_lr,
                          warmup_steps=opt.warmup_step, decay_steps=200000)

        _, config = load_train_net(opt)
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=1024.0, scale_factor=2, scale_window=1000)

        # Backbone Part
        backbone_eval_net = backbone_base_net = PanguAlphaModel(config)
        backbone_train_net = BackboneNecessrayLossNet(backbone_base_net)
        backbone_with_loss = _VirtualDatasetCell(backbone_train_net)
        backbone_params = backbone_with_loss.trainable_params()
        backbone_group_params = set_weight_decay(backbone_params)
        backbone_optim_inst = FP32StateAdamWeightDecay(backbone_group_params, lr, eps=1e-8, beta1=0.9, beta2=0.95)
        backbone_optim = PanguAlphaAdam(backbone_train_net, backbone_optim_inst, update_cell, config, backbone_yaml)

        self.backbone_fl_model = FLModel(yaml_data=backbone_yaml, network=backbone_train_net,
                                         eval_network=backbone_eval_net, optimizers=backbone_optim)

        # load ckpt
        if opt.resume:
            self.backbone_fl_model.load_ckpt()

        # get compress config
        compress_configs = self.backbone_fl_model.get_compress_configs()

        # build vertical communicator
        http_server_config = ServerConfig(server_name='follower', server_address=opt.http_server_address)
        remote_server_config = ServerConfig(server_name='leader', server_address=opt.remote_server_address)

        self.vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
                                                                   remote_server_config=remote_server_config,
                                                                   compress_configs=compress_configs)
        self.vertical_communicator.launch()

    def start(self):
        """
        Run follower trainer
        """

        step = 0
        while True:
            embedding_out = self.vertical_communicator.receive("leader")
            backbone_out = self.backbone_fl_model.forward_one_step(remote_data_batch=embedding_out)
            self.vertical_communicator.send_tensors("leader", backbone_out)
            head_scale = self.vertical_communicator.receive("leader")

            backbone_scale = self.backbone_fl_model.backward_one_step(remote_data_batch=embedding_out, sens=head_scale)
            backbone_scale['hidden_states'].pop('attention_mask')
            self.vertical_communicator.send_tensors("leader", backbone_scale)

            step += 1
            if step % 1000 == 0:
                self.backbone_fl_model.save_ckpt()


if __name__ == '__main__':
    device_num = 1
    rank_id = 0
    opt = get_args()
    set_parse(opt)

    logging.basicConfig(filename='follower_process.log', level=logging.INFO)
    logging.info("config is:")
    logging.info(opt)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    follower_trainer = FollowerTrainer()
    follower_trainer.start()
