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
"""Local splitnn of wide and deep on criteo dataset."""
import os
import logging

from mindspore import context, Tensor
from mindspore.train.summary import SummaryRecord

from mindspore_federated import FLModel, FLYamlData

from criteo_dataset import create_dataset, DataType
from network_config import config
from wide_and_deep import LeaderNet, LeaderLossNet, LeaderEvalNet, \
    FollowerNet, FollowerLossNet, AUCMetric


def construct_local_dataset():
    """create dataset object according to config info."""
    path = config.data_path
    train_bs = config.batch_size
    eval_bs = config.batch_size
    if config.dataset_type == "tfrecord":
        ds_type = DataType.TFRECORD
    elif config.dataset_type == "mindrecord":
        ds_type = DataType.MINDRECORD
    else:
        ds_type = DataType.H5

    train_dataset = create_dataset(path, batch_size=train_bs, data_type=ds_type)
    eval_dataset = create_dataset(path, train_mode=False, batch_size=eval_bs, data_type=ds_type)
    return train_dataset, eval_dataset


if __name__ == '__main__':
    logging.basicConfig(filename='log_local_{}.txt'.format(config.device_target), level=logging.INFO)
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    leader_yaml_data = FLYamlData(config.leader_yaml_path)
    follower_yaml_data = FLYamlData(config.follower_yaml_path)
    # local data iteration for experiment
    ds_train, ds_eval = construct_local_dataset()
    train_iter = ds_train.create_dict_iterator()
    eval_iter = ds_eval.create_dict_iterator()
    train_size = ds_train.get_dataset_size()
    eval_size = ds_eval.get_dataset_size()
    # Leader Part
    leader_base_net = LeaderNet(config)
    leader_train_net = LeaderLossNet(leader_base_net, config)
    leader_eval_net = LeaderEvalNet(leader_base_net)
    eval_metric = AUCMetric()
    leader_fl_model = FLModel(yaml_data=leader_yaml_data,
                              network=leader_train_net,
                              metrics=eval_metric,
                              eval_network=leader_eval_net)

    # Follower Part
    follower_eval_net = follower_base_net = FollowerNet(config)
    follower_train_net = FollowerLossNet(follower_base_net, config)
    follower_fl_model = FLModel(yaml_data=follower_yaml_data,
                                network=follower_train_net,
                                eval_network=follower_eval_net)

    # resume if you have pretrained checkpoint file
    if config.resume:
        if os.path.exists(config.pre_trained_follower):
            follower_fl_model.load_ckpt(path=config.pre_trained_follower)
        if os.path.exists(config.pre_trained_leader):
            leader_fl_model.load_ckpt(path=config.pre_trained_leader)

    # forward/backward batch by batch
    steps_per_epoch = ds_train.get_dataset_size()
    with SummaryRecord('./summary') as summary_record:
        for epoch in range(config.epochs):
            for step, item in enumerate(train_iter, start=1):
                step = steps_per_epoch * epoch + step
                follower_out = follower_fl_model.forward_one_step(item)
                leader_out = leader_fl_model.forward_one_step(item, follower_out)
                scale = leader_fl_model.backward_one_step(item, follower_out)
                follower_fl_model.backward_one_step(item, sens=scale)
                if step % 100 == 0:
                    summary_record.add_value('scalar', 'wide_loss', leader_out['wide_loss'])
                    summary_record.add_value('scalar', 'deep_loss', leader_out['deep_loss'])
                    summary_record.record(step)
                    logging.info('epoch %d step %d/%d wide_loss: %f deep_loss: %f',
                                 epoch, step, train_size, leader_out['wide_loss'], leader_out['deep_loss'])

                    # save checkpoint
                    leader_fl_model.save_ckpt()
                    follower_fl_model.save_ckpt()

                    for eval_item in eval_iter:
                        follower_out = follower_fl_model.forward_one_step(eval_item)
                        leader_eval_out = leader_fl_model.eval_one_step(eval_item, follower_out)
                    auc = eval_metric.eval()
                    eval_metric.clear()
                    summary_record.add_value('scalar', 'auc', Tensor(auc))
                    logging.info('----evaluation---- epoch %d auc %f', epoch, auc)
