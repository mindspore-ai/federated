# Copyright 2023 Huawei Technologies Co., Ltd
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
"""start get cross silo loss"""

import time
import numpy as np
from mindspore import Callback, Tensor


def try_loss(losses):
    """get loss after an iteration"""
    loss = losses
    if isinstance(loss, (tuple, list)):
        if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
            loss = loss[0]

    if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
        loss = np.mean(loss.asnumpy())

    if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
        raise ValueError("Invalid loss, terminating training.")

    return loss

def try_preloss(prelosses):
    """get preloss after an iteration"""
    preloss = prelosses

    if isinstance(preloss, (tuple, list)):
        if isinstance(preloss[0], Tensor) and isinstance(preloss[0].asnumpy(), np.ndarray):
            preloss = preloss[0]

    if isinstance(preloss, Tensor) and isinstance(preloss.asnumpy(), np.ndarray):
        preloss = np.mean(preloss.asnumpy())

    if isinstance(preloss, float) and (np.isnan(preloss) or np.isinf(preloss)):
        raise ValueError("Invalid preloss, terminating training.")

    return preloss


class LossGet(Callback):
    """# define loss callback for packaged network"""

    def __init__(self, per_print_times, data_size):
        super(LossGet, self).__init__()
        self._per_step_mseconds = None
        self.epoch_time = None
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self._loss = 0.0
        self.data_size = data_size
        self.loss_list = []
        self._preloss = 0.0
        self.preloss_list = []

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        preloss = cb_params.preloss

        loss = try_loss(loss)
        preloss = try_preloss(preloss)

        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            self._loss = loss
            self.loss_list.append(loss)

        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            self._preloss = preloss
            self.preloss_list.append(preloss)

    def epoch_begin(self, _):
        """define epoch begin func"""
        self.epoch_time = time.time()


    def epoch_end(self, _):
        """define epoch end func"""
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        self._per_step_mseconds = epoch_mseconds / self.data_size

    def get_loss(self):
        """get loss"""
        return self.loss_list

    def get_preloss(self):
        """get loss"""
        return self.preloss_list

    def get_per_step_time(self):
        """get per step time"""
        return self._per_step_mseconds
