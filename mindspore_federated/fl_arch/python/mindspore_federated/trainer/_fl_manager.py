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
"""FederatedLearningManager related class and functions."""

from copy import deepcopy
import numpy as np
from mindspore import nn
from mindspore import load_param_into_net
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.train.callback import Callback
from mindspore._checkparam import Validator, Rel
from mindspore_federated._mindspore_federated import Federated_, FLContext
from mindspore_federated import log as logger

from ..startup.ssl_config import init_ssl_config
from ..startup.yaml_config import load_yaml_config
from ..common import _fl_context, check_type

ONE_STEP_PER_ITERATION = 1
TRAIN_BEGIN_STEP_NUM = 1
TRAIN_END_STEP_NUM = 0


class _StartFLJob:
    """
    StartFLJob for Federated Learning Worker.
    """

    def __init__(self, data_size):
        self._data_size = data_size

    def construct(self):
        return Federated_.start_fl_job(self._data_size)


class _UpdateAndGetModel:
    """
    Update and Get Model for Federated Learning Worker.
    """

    def __init__(self, weights):
        super(_UpdateAndGetModel, self).__init__()
        self._weights = weights

    def construct(self):
        return Federated_.update_and_get_model(self._weights)


class _ExchangeKeys:
    """
    Exchange Keys for Stable PW Encrypt.
    """

    @staticmethod
    def construct():
        return Federated_.exchange_keys()


class _GetKeys:
    """
    Get Keys for Stable PW Encrypt.
    """

    @staticmethod
    def construct():
        return Federated_.get_keys()


class _PullWeight:
    """
    Pull Weight for Federated Learning Worker.
    """

    def __init__(self, pull_weight_params):
        self.pull_weight_params = pull_weight_params

    def construct(self):
        return Federated_.pull_weight(self.pull_weight_params)


class _PushWeight:
    """
    Push Weight for Federated Learning Worker.
    """

    def __init__(self, weights):
        self._weights = weights

    def construct(self):
        return Federated_.push_weight(self._weights)


class PushMetrics:
    """
    Push Metrics for Federated Learning Worker.
    """

    @staticmethod
    def construct(loss, accuracy):
        return Federated_.push_metrics(loss, accuracy)


def _get_fl_param_names(network, fl_param_names, requires_aggr=False):
    for sub_cell in network.cells():
        fl_param_names = _get_fl_param_names(sub_cell, fl_param_names, requires_aggr)
        if isinstance(sub_cell, nn.Optimizer):
            for k in sub_cell.parameters:
                if requires_aggr and not k.requires_aggr:
                    continue
                if k.name not in fl_param_names:
                    fl_param_names.append(k.name)
    return fl_param_names


class FederatedLearningManager(Callback):
    """
    Manage Federated Learning during training.

    Args:
        yaml_config (str): The yaml file path. More detail see `federated_server_yaml <https://gitee.com/mindspore/federated/blob/master/docs/api/api_python_en/horizontal/federated_server_yaml.md>`_.
        model (nn.Cell): A model for Federated Training.
        sync_frequency (int): Synchronization frequency of parameters in Federated Learning.
                              Note that in dataset sink mode, the unit of the frequency is the number of epochs.
                              Otherwise, the unit of the frequency is the number of steps.
                              The initial synchronization frequency is in adaptive synchronization frequency mode and
                              the synchronization frequency in fixed frequency mode.
        http_server_address (str): The http server address used for communicating. Default: "".
        data_size (int): The data size to be reported to the worker. Default: 1.
        sync_type (str): The synchronization type of parameter in Federated Learning.
                         Supports ["fixed", "adaptive"]. Default: "fixed".

                         - fixed: The frequency of parameter synchronization is fixed.

                         - adaptive: The frequency of parameter synchronization changes adaptively.

        ssl_config (Union(None, SSLConfig)): Config of ssl. Default: None.
        min_consistent_rate (float): Minimum consistency ratio threshold. The greater the value, the more
                                     difficult it is to improve the synchronization frequency.
                                     Value range: greater than or equal to 0.0. Default: 1.1.
        min_consistent_rate_at_round (int): The number of rounds of the minimum consistency ratio threshold.
                                            The greater the value, the more difficult it is to improve the
                                            synchronization frequency.
                                            Value range: greater than or equal to 0. Default: 0.
        ema_alpha (float): Gradient consistency smoothing coefficient. The smaller the value, the more the
                           frequency will be judged according to the gradient bifurcation of the current round
                           more. Otherwise it will be judged according to the historical gradient bifurcation
                           more.
                           Value range: (0.0, 1.0). Default: 0.5.
        observation_window_size (int): The number of rounds in the observation time window. The greater the
                                       value, the more difficult it is to reduce the synchronization frequency.
                                       Value range: greater than 0. Default: 5.
        frequency_increase_ratio (int): Frequency increase amplitude. The greater the value, the greater the
                                        frequency increase amplitude.
                                        Value range: greater than 0. Default: 2.
        unchanged_round (int): The number of rounds whose frequency does not change. The frequency is unchanged
                               before unchanged_round rounds.
                               Value range: greater than or equal to 0. Default: 0.

    Examples:
        >>> from mindspore_federated import FederatedLearningManager
        >>> from mindspore import nn, Model
        >>> from network.lenet import LeNet5, create_dataset_from_folder
        >>> network = LeNet5(62, 3)
        >>> federated_learning_manager = FederatedLearningManager(
        ...     yaml_config="default_yaml_config.yaml",
        ...     model=network,
        ...     sync_frequency=100,
        ...     http_server_address="127.0.0.1:10086",
        ... )
        >>> net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> net_opt = nn.Momentum(network.trainable_params(), 0.001, 0.9)
        >>> model = Model(network, net_loss, net_opt)
        >>> dataset = create_dataset_from_folder("0/train/", 32, 16, 1)
        >>> model.train(100, dataset, callbacks=[federated_learning_manager], dataset_sink_mode=False)
    """

    def __init__(self, yaml_config, model, sync_frequency, http_server_address="", data_size=1, sync_type='fixed',
                 ssl_config=None, **kwargs):
        super(FederatedLearningManager, self).__init__()
        check_type.check_str("yaml_config", yaml_config)
        init_ssl_config(ssl_config)
        load_yaml_config(yaml_config, _fl_context.ROLE_OF_SERVER)

        ctx = FLContext.get_instance()
        server_mode = ctx.server_mode()
        encrypt_type = ctx.encrypt_type()
        ctx.set_http_server_address(http_server_address)

        Federated_.init_federated_worker()

        Validator.check_isinstance('model', model, nn.Cell)
        Validator.check_positive_int(sync_frequency)
        Validator.check_string(sync_type, ["fixed", "adaptive"])
        self._server_mode = server_mode
        self._model = model
        self._sync_frequency = sync_frequency
        self._next_sync_iter_id = self._sync_frequency
        self._data_size = data_size
        self._sync_type = sync_type
        self._global_step = 0
        self._encrypt_type = encrypt_type
        if self._encrypt_type not in (
                _fl_context.ENCRYPT_NONE,
                _fl_context.ENCRYPT_STABLE_PW) and self._server_mode == _fl_context.SERVER_MODE_CLOUD:
            raise ValueError(
                "encrypt_mode must be 'NOT_ENCRYPT' or 'STABLE_PW_ENCRYPT', but got {}.".format(self._encrypt_type))
        if self._is_adaptive_sync():
            self._as_set_init_state(kwargs)
            self._as_wrap_cell()
        logger.info(f"Step number needs to run per iteration {self._next_sync_iter_id},"
                    f"server mode {self._server_mode}, encrypt type {self._encrypt_type},"
                    f"http server address {http_server_address}")

    def __del__(self):
        Federated_.stop_federated_worker()

    def _is_adaptive_sync(self):
        """
        Determine whether adaptive frequency synchronization is required.
        """
        return self._sync_type == "adaptive"

    def _as_set_init_state(self, kwargs):
        """
        Setting the initial state for adaptive synchronization.
        """
        self._as_prefix = "as_abs_grad."

        self._min_consistent_rate = kwargs.get("min_consistent_rate", 1.1)
        Validator.check_non_negative_float(self._min_consistent_rate)
        self._min_consistent_rate_at_round = kwargs.get("min_consistent_rate_at_round", 0)
        Validator.check_non_negative_int(self._min_consistent_rate_at_round)
        self._ema_alpha = kwargs.get("ema_alpha", 0.5)
        Validator.check_float_range(self._ema_alpha, 0.0, 1.0, Rel.INC_NEITHER)
        self._observation_window_size = kwargs.get("observation_window_size", 5)
        Validator.check_positive_int(self._observation_window_size)
        self._frequency_increase_ratio = kwargs.get("frequency_increase_ratio", 2)
        Validator.check_positive_int(self._frequency_increase_ratio)
        self._unchanged_round = kwargs.get("unchanged_round", 0)
        Validator.check_non_negative_int(self._unchanged_round)

        self._round_id = 0
        self._last_param = {_.name: deepcopy(_.asnumpy()) for _ in self._model.trainable_params()
                            if self._as_prefix not in _.name}
        self._model_size = 0
        self._grads_ema = dict()
        self._abs_grads_ema = dict()
        for param in self._model.trainable_params():
            if self._as_prefix not in param.name:
                self._model_size += np.product(param.shape)
                self._grads_ema[param.name] = np.zeros(param.shape)
                self._abs_grads_ema[param.name] = np.zeros(param.shape)
        self._model_size = float(self._model_size)

    def _as_wrap_cell(self):
        """
        Wrap Cell for adaptive synchronization.
        """
        param_list = list()
        for param in self._model.trainable_params():
            new_param = param.clone()
            new_param.name = self._as_prefix + param.name
            param_list.append(new_param)
        for param in param_list:
            self._model.insert_param_to_cell(param.name, param, False)

    def _as_set_grads(self):
        """
        Set the absolute value of the gradient for adaptive synchronization.
        """
        abs_grads = dict()
        for param in self._model.trainable_params():
            if self._as_prefix not in param.name:
                abs_grads[self._as_prefix + param.name] = np.abs(param.asnumpy() - self._last_param[param.name])
        for param in self._model.trainable_params():
            if self._as_prefix in param.name:
                param.set_data(Parameter(abs_grads[param.name]))

    def _as_analyze_gradient(self):
        """
        Analysis of relevant statistics based on gradient for adaptive synchronization.
        """
        ctx = FLContext.get_instance()
        worker_num = int(ctx.start_fl_job_threshold() * ctx.update_model_ratio())
        ema_alpha = self._ema_alpha
        consistent_rate_sum = 0.0
        grads = dict()
        abs_grads = dict()
        for param in self._model.trainable_params():
            if self._as_prefix in param.name:
                abs_grads[param.name.replace(self._as_prefix, '')] = param.asnumpy() * worker_num
            else:
                grads[param.name] = (param.asnumpy() - self._last_param[param.name]) * worker_num
        for last_p in self._last_param:
            self._grads_ema[last_p] = ema_alpha * self._grads_ema[last_p] + (1 - ema_alpha) * grads[last_p]
            self._abs_grads_ema[last_p] = ema_alpha * self._abs_grads_ema[last_p] + (1 - ema_alpha) * abs_grads[last_p]
            divide_base = np.where(self._abs_grads_ema[last_p] == 0,
                                   np.ones(self._abs_grads_ema[last_p].shape), self._abs_grads_ema[last_p])
            layer_consistent_rate = np.abs(self._grads_ema[last_p]) / divide_base
            consistent_rate_sum += np.sum(layer_consistent_rate)

        consistent_rate = float(consistent_rate_sum / self._model_size)

        if self._min_consistent_rate > consistent_rate:
            self._min_consistent_rate = consistent_rate
            self._min_consistent_rate_at_round = self._round_id
        elif self._round_id - self._min_consistent_rate_at_round > self._observation_window_size and \
                self._sync_frequency > 1 and self._round_id > self._unchanged_round:
            self._sync_frequency = (self._sync_frequency + self._frequency_increase_ratio - 1) \
                                   // self._frequency_increase_ratio
            self._min_consistent_rate = 1.1
            self._min_consistent_rate_at_round = self._round_id
            self._observation_window_size *= self._frequency_increase_ratio

            for param in self._model.trainable_params():
                if self._as_prefix not in param.name:
                    self._grads_ema[param.name] = np.zeros(param.shape)
                    self._abs_grads_ema[param.name] = np.zeros(param.shape)

    def _as_set_last_param(self):
        """
        Set the value of last parameters for adaptive synchronization.
        """
        self._last_param = {_.name: deepcopy(_.asnumpy()) for _ in self._model.trainable_params()
                            if self._as_prefix not in _.name}

    def _start_pull_weight(self):
        """
        Pull weight from server in hybrid training mode.
        """
        logger.info("Try to pull weights. Local step number: {}".format(self._global_step))
        # The worker has to train self._sync_frequency standalone iterations before it communicates with server.
        if self._global_step % self._sync_frequency != TRAIN_BEGIN_STEP_NUM:
            return

        pull_weight_params = list()
        pull_weight_params = _get_fl_param_names(self._model, pull_weight_params, True)
        if not pull_weight_params:
            pull_weight_params = [_.name for _ in self._model.trainable_params()]
        weight_infos = {}
        for param in self._model.trainable_params():
            if param.name not in pull_weight_params:
                continue
            param_np = param.asnumpy()
            if param_np.dtype != np.float32:
                continue
            weight_infos[param.name] = (param_np.shape, param_np.dtype)

        pull_weight = _PullWeight(pull_weight_params)
        weights = pull_weight.construct()
        if not weights:
            raise ValueError("Weights from pulling weight is empty!")
        parameter_dict = {}
        for key, value in weights.items():
            if key not in weight_infos:
                continue
            shape, dtype = weight_infos[key]
            param_data = np.reshape(value, shape).astype(dtype)
            parameter_dict[key] = Parameter(Tensor(param_data), name=key)
        load_param_into_net(self._model, parameter_dict)

    def _start_push_weight(self):
        """
        Push weight to server in hybrid training mode.
        """
        logger.info("Try to push weights. Local step number: {}".format(self._global_step))
        if self._global_step % self._sync_frequency != TRAIN_END_STEP_NUM:
            return

        push_weight_params = list()
        push_weight_params = _get_fl_param_names(self._model, push_weight_params, True)
        if not push_weight_params:
            push_weight_params = [_.name for _ in self._model.trainable_params()]
        weights = dict()
        for param in self._model.trainable_params():
            if param.name not in push_weight_params:
                continue
            weight = param.asnumpy().reshape(-1).tolist()
            weights[param.name] = weight
        push_weight = _PushWeight(weights)
        push_weight.construct()

    def step_end(self, run_context):
        """
        Synchronization parameters at the end of step. If sync_type is "adaptive", the synchronous frequency is
        adaptively adjusted here.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self._global_step += 1
        if self._server_mode == _fl_context.SERVER_MODE_CLOUD:
            if self._global_step == self._next_sync_iter_id:
                start_fl_job = _StartFLJob(self._data_size)
                start_fl_job.construct()
                if self._is_adaptive_sync():
                    self._as_set_grads()
                if self._encrypt_type == _fl_context.ENCRYPT_STABLE_PW:
                    exchange_keys = _ExchangeKeys()
                    exchange_keys.construct()
                    get_keys = _GetKeys()
                    get_keys.construct()

                weights = {}
                weight_infos = {}
                for param in self._model.trainable_params():
                    param_np = param.asnumpy()
                    if param_np.dtype != np.float32:
                        continue
                    weight_infos[param.name] = (param_np.shape, param_np.dtype)
                    weights[param.name] = param_np.reshape(-1).tolist()
                update_and_get_model = _UpdateAndGetModel(weights)
                feature_map = update_and_get_model.construct()
                if not feature_map:
                    raise ValueError("Feature map from getting model is empty!")
                parameter_dict = {}
                for key, value in feature_map.items():
                    if key not in weight_infos:
                        continue
                    shape, dtype = weight_infos[key]
                    param_data = np.reshape(value, shape).astype(dtype)
                    parameter_dict[key] = Parameter(Tensor(param_data), name=key)
                load_param_into_net(self._model, parameter_dict)
                logger.info("Load params from getting model into net, global step is {}.".format(self._global_step))
                self._next_sync_iter_id = self._global_step + self._sync_frequency
                if self._is_adaptive_sync():
                    self._as_analyze_gradient()
                    self._round_id += 1
                    self._as_set_last_param()
                cb_params = run_context.original_args()
                logger.info(
                    "total epoch num:{}, batch num:{}, Current epoch num is: {}, Current step num is: {}".format(
                        cb_params.epoch_num, cb_params.batch_num, cb_params.cur_epoch_num,
                        cb_params.cur_step_num))
        elif self._server_mode == _fl_context.SERVER_MODE_HYBRID:
            self._start_pull_weight()
            self._start_push_weight()
