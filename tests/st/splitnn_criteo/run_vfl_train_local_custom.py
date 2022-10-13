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

import logging

from mindspore import context, Tensor
from mindspore.train.summary import SummaryRecord

from mindspore_federated import FLModel, vfl_utils

from run_vfl_train_local import construct_local_dataset
from network_config import config
from network.wide_and_deep import AUCMetric
from network.wide_and_deep_custom import LeaderNet, LeaderLossNet, LeaderGradNet, LeaderEvalNet, \
    FollowerNet, FollowerLossNet


if __name__ == '__main__':
    logging.basicConfig(filename='log_local_{}_custom.txt'.format(config.device_target), level=logging.INFO)
    context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    leader_yaml_data, leader_fp = vfl_utils.parse_yaml_file(config.leader_yaml_path)
    follower_yaml_data, follower_fp = vfl_utils.parse_yaml_file(config.follower_yaml_path)
    # local data iteration for experiment
    ds_train, ds_eval = construct_local_dataset()
    train_iter = ds_train.create_dict_iterator()
    eval_iter = ds_eval.create_dict_iterator()
    train_size = ds_train.get_dataset_size()
    eval_size = ds_eval.get_dataset_size()
    # Leader Part
    leader_base_net = LeaderNet(config)
    leader_train_net = LeaderLossNet(leader_base_net, config)
    leader_grad_net = LeaderGradNet(leader_base_net, config)
    leader_eval_net = LeaderEvalNet(leader_base_net)
    eval_metric = AUCMetric()
    leader_fl_model = FLModel(role='leader',
                              network=leader_base_net,
                              train_network=leader_train_net,
                              grad_network=leader_grad_net,
                              metrics=eval_metric,
                              eval_network=leader_eval_net,
                              yaml_data=leader_yaml_data)

    # Follower Part
    follower_eval_net = follower_base_net = FollowerNet(config)
    follower_train_net = FollowerLossNet(follower_base_net, config)
    follower_fl_model = FLModel(role='follower',
                                network=follower_base_net,
                                train_network=follower_train_net,
                                eval_network=follower_eval_net,
                                yaml_data=follower_yaml_data)
    # forward/backward batch by batch
    eval_metric = AUCMetric()
    with SummaryRecord('./summary') as summary_record:
        for epoch in range(config.epochs):
            for step, item in enumerate(train_iter, start=1):
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
            for eval_item in eval_iter:
                follower_out = follower_fl_model.forward_one_step(eval_item)
                leader_eval_out = leader_fl_model.eval_one_step(eval_item, follower_out, eval_metric)
            auc = eval_metric.eval()
            eval_metric.clear()
            summary_record.add_value('scalar', 'auc', Tensor(auc))
            logging.info('----evaluation---- epoch %d auc %f', epoch, auc)
    leader_fp.close()
    follower_fp.close()
