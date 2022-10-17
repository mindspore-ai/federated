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

from mindspore import context, Tensor
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.train.summary import SummaryRecord
from mindspore_federated import FLModel, FLYamlData

from src.split_pangu_alpha import PanguAlphaModel, BackboneLossNet, PanGuHead, HeadLossNet, EmbeddingLayer, \
    EmbeddingLossNet, PPLMetric

from src.utils import LearningRate, get_args, construct_local_dataset, load_train_net, set_weight_decay, \
    set_embedding_weight_decay
from src.pangu_optim import PanguAlphaAdam, FP32StateAdamWeightDecay
from src.pangu_alpha_config import set_parse


project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")


if __name__ == '__main__':
    device_num = 1
    rank_id = 0
    print("rank_id is {}, device_num is {}".format(rank_id, device_num))
    opt = get_args()
    set_parse(opt)
    logging.basicConfig(filename='splitnn_pangu_local.txt', level=logging.INFO)
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    # read, parse and check the .yaml files of sub-networks
    embedding_yaml = FLYamlData('./embedding.yaml')
    backbone_yaml = FLYamlData('./backbone.yaml')
    head_yaml = FLYamlData('./head.yaml')

    # local data iteration for experiment
    ds_train = construct_local_dataset(opt, rank_id, device_num)
    train_iter = ds_train.create_dict_iterator()
    train_size = ds_train.get_dataset_size()
    # eval data iteration for example
    ds_eval = construct_local_dataset(opt, rank_id, device_num, is_training=False)
    eval_iter = ds_eval.create_dict_iterator()
    eval_size = ds_eval.get_dataset_size()

    # loss scale
    lr = LearningRate(learning_rate=opt.start_lr, end_learning_rate=opt.end_lr,
                      warmup_steps=opt.warmup_step, decay_steps=200000)

    loss, config = load_train_net(opt)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=1024.0, scale_factor=2, scale_window=1000)
    eval_metric = PPLMetric(opt.seq_length)
    # Embedding/Tail Part
    embedding_base_net = EmbeddingLayer(config)
    embedding_eval_net = embedding_train_net = EmbeddingLossNet(embedding_base_net, config)
    embedding_with_loss = _VirtualDatasetCell(embedding_eval_net)
    embedding_params = embedding_with_loss.trainable_params()
    embedding_group_params = set_embedding_weight_decay(embedding_params)
    embedding_optim_inst = FP32StateAdamWeightDecay(embedding_group_params, lr, eps=1e-8, beta1=0.9, beta2=0.95)
    embedding_optim = PanguAlphaAdam(embedding_train_net, embedding_optim_inst, update_cell, config, embedding_yaml)
    # Backbone Part
    backbone_eval_net = backbone_base_net = PanguAlphaModel(config)
    backbone_train_net = BackboneLossNet(backbone_base_net)
    backbone_with_loss = _VirtualDatasetCell(backbone_train_net)
    backbone_params = backbone_with_loss.trainable_params()
    backbone_group_params = set_weight_decay(backbone_params)
    backbone_optim_inst = FP32StateAdamWeightDecay(backbone_group_params, lr, eps=1e-8, beta1=0.9, beta2=0.95)
    backbone_optim = PanguAlphaAdam(backbone_train_net, backbone_optim_inst, update_cell, config, backbone_yaml)
    # Head Party
    head_base_net = PanGuHead(config)
    head_eval_net = head_train_net = HeadLossNet(head_base_net, loss, config)
    head_with_loss = _VirtualDatasetCell(head_train_net)
    head_params = head_with_loss.trainable_params()
    head_group_params = set_weight_decay(head_params)
    head_optim_inst = FP32StateAdamWeightDecay(head_group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95)
    head_optim = PanguAlphaAdam(head_train_net, head_optim_inst, update_cell, config, head_yaml)

    # FLModel definition
    head_fl_model = FLModel(yaml_data=head_yaml,
                            network=head_base_net,
                            train_network=head_train_net,
                            eval_network=head_eval_net,
                            optimizers=head_optim,
                            metrics=eval_metric)
    backbone_fl_model = FLModel(yaml_data=backbone_yaml,
                                network=backbone_base_net,
                                train_network=backbone_train_net,
                                eval_network=backbone_eval_net,
                                optimizers=backbone_optim)
    embedding_fl_model = FLModel(yaml_data=embedding_yaml,
                                 network=embedding_base_net,
                                 train_network=embedding_train_net,
                                 eval_network=embedding_eval_net,
                                 optimizers=embedding_optim)

    # resume if you have checkpoint file or dir
    # embedding_fl_model.load_ckpt()
    # backbone_fl_model.load_ckpt()
    # head_fl_model.load_ckpt()

    # forward/backward batch by batch
    with SummaryRecord('./summary') as summary_record:
        for epoch in range(50):
            for step, item in enumerate(train_iter, start=1):
                # forward process
                step = epoch * train_size + step
                embedding_out = embedding_fl_model.forward_one_step(item)
                backbone_out = backbone_fl_model.forward_one_step(item, embedding_out)
                logit_out = head_fl_model.forward_one_step(item, backbone_out)
                # backward process
                head_scale = head_fl_model.backward_one_step(item, backbone_out)
                backbone_scale = backbone_fl_model.backward_one_step(item, embedding_out, sens=head_scale)
                embedding_fl_model.backward_one_step(item, sens=backbone_scale)
                if step % 10 == 0:
                    summary_record.add_value('scalar', 'output', logit_out['output'])
                    summary_record.record(step)
                    logging.info('epoch %d step %d/%d loss: %f', epoch, step, train_size, logit_out['output'])

                    # save checkpoint
                    embedding_fl_model.save_ckpt()
                    backbone_fl_model.save_ckpt()
                    head_fl_model.save_ckpt()

                    for eval_item in eval_iter:
                        # forward process
                        embedding_out = embedding_fl_model.forward_one_step(item)
                        backbone_out = backbone_fl_model.forward_one_step(item, embedding_out)
                        logit_out = head_fl_model.eval_one_step(item, backbone_out)

                    ppl = eval_metric.eval()
                    eval_metric.clear()
                    summary_record.add_value('scalar', 'ppl', Tensor(ppl))
                    logging.info('----evaluation---- epoch %d step %d ppl %f', epoch, step, ppl)
