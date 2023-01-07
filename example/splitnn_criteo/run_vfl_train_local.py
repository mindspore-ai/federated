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
from collections import OrderedDict

from mindspore import context, Tensor
from mindspore.train.summary import SummaryRecord

from mindspore_federated import FLModel, FLYamlData

from criteo_dataset import create_dataset, DataType
from network_config import config
from wide_and_deep import LeaderTopNet, LeaderTopLossNet, LeaderTopEvalNet, LeaderBottomNet, LeaderBottomLossNet, \
    FollowerBottomNet, FollowerBottomLossNet, AUCMetric


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


def apply_embedding_dp(fl_embedding, embedding_dp):
    fl_embedding['follower_wide_embedding'] = embedding_dp(fl_embedding['follower_wide_embedding'])
    fl_embedding['follower_deep_embedding'] = embedding_dp(fl_embedding['follower_deep_embedding'])
    return fl_embedding


if __name__ == '__main__':
    logging.basicConfig(filename='log_local_{}.txt'.format(config.device_target), level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    leader_top_yaml_data = FLYamlData(config.leader_top_yaml_path)
    leader_bottom_yaml_data = FLYamlData(config.leader_bottom_yaml_path)
    follower_bottom_yaml_data = FLYamlData(config.follower_bottom_yaml_path)
    # local data iteration for experiment
    ds_train, ds_eval = construct_local_dataset()
    train_iter = ds_train.create_dict_iterator()
    eval_iter = ds_eval.create_dict_iterator()
    train_size = ds_train.get_dataset_size()
    eval_size = ds_eval.get_dataset_size()
    # Leader Bottom Net
    leader_bottom_eval_net = leader_bottom_base_net = LeaderBottomNet(config)
    leader_bottom_train_net = LeaderBottomLossNet(leader_bottom_base_net, config)
    leader_bottom_fl_model = FLModel(yaml_data=leader_bottom_yaml_data,
                                     network=leader_bottom_train_net,
                                     eval_network=leader_bottom_eval_net)
    # Leader Top Net
    leader_top_base_net = LeaderTopNet(config)
    leader_top_train_net = LeaderTopLossNet(leader_top_base_net, config)
    leader_top_eval_net = LeaderTopEvalNet(leader_top_base_net)
    eval_metric = AUCMetric()
    leader_top_fl_model = FLModel(yaml_data=leader_top_yaml_data,
                                  network=leader_top_train_net,
                                  metrics=eval_metric,
                                  eval_network=leader_top_eval_net)

    # Follower Bottom Net
    follower_bottom_eval_net = follower_bottom_base_net = FollowerBottomNet(config)
    follower_bottom_train_net = FollowerBottomLossNet(follower_bottom_base_net, config)
    follower_bottom_fl_model = FLModel(yaml_data=follower_bottom_yaml_data,
                                       network=follower_bottom_train_net,
                                       eval_network=follower_bottom_eval_net)

    # resume if you have pretrained checkpoint file
    if config.resume:
        if os.path.exists(config.pre_trained_follower_bottom):
            follower_bottom_fl_model.load_ckpt(path=config.pre_trained_follower_bottom)
        if os.path.exists(config.pre_trained_leader_bottom):
            leader_bottom_fl_model.load_ckpt(path=config.pre_trained_leader_bottom)
        if os.path.exists(config.pre_trained_leader_top):
            leader_top_fl_model.load_ckpt(path=config.pre_trained_leader_top)

    # forward/backward batch by batch
    steps_per_epoch = ds_train.get_dataset_size()
    with SummaryRecord('./summary') as summary_record:
        for epoch in range(config.epochs):
            for step, item in enumerate(train_iter, start=1):
                step = steps_per_epoch * epoch + step
                follower_embedding = follower_bottom_fl_model.forward_one_step(item)
                if follower_bottom_fl_model.embedding_dp:
                    follower_embedding = apply_embedding_dp(follower_embedding, follower_bottom_fl_model.embedding_dp)
                leader_embedding = leader_bottom_fl_model.forward_one_step(item)
                item.update(leader_embedding)
                leader_out = leader_top_fl_model.forward_one_step(item, follower_embedding)
                scale = leader_top_fl_model.backward_one_step(item, follower_embedding)
                leader_scale = {'loss': OrderedDict(list(scale['loss'].items())[:2])}
                follower_scale = {'loss': OrderedDict(list(scale['loss'].items())[2:])}
                leader_bottom_fl_model.backward_one_step(item, sens=leader_scale)
                follower_bottom_fl_model.backward_one_step(item, sens=follower_scale)
                if step % 100 == 0:
                    summary_record.add_value('scalar', 'loss', leader_out['loss'])
                    summary_record.record(step)
                    logging.info('epoch %d step %d/%d loss: %f',
                                 epoch, step, train_size, leader_out['loss'])

                    # save checkpoint
                    leader_top_fl_model.save_ckpt()
                    leader_bottom_fl_model.save_ckpt()
                    follower_bottom_fl_model.save_ckpt()

            for eval_item in eval_iter:
                follower_embedding = follower_bottom_fl_model.forward_one_step(eval_item)
                leader_embedding = leader_bottom_fl_model.forward_one_step(eval_item)
                embedding = follower_embedding
                embedding.update(leader_embedding)
                leader_eval_out = leader_top_fl_model.eval_one_step(eval_item, embedding)
            auc = eval_metric.eval()
            eval_metric.clear()
            summary_record.add_value('scalar', 'auc', Tensor(auc))
            logging.info('----evaluation---- epoch %d auc %f', epoch, auc)
