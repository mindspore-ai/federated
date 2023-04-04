# pylint: disable=missing-docstring
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
"""FederatedLearningManager related class and functions."""

from copy import deepcopy
import numpy as np
import mindspore.ops as ops
from mindspore import nn
from mindspore.nn import Cell
from mindspore import load_param_into_net
from mindspore.communication.management import init, get_rank
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.train.callback import Callback
from mindspore_federated.common import _checkparam as validator
from mindspore_federated._mindspore_federated import Federated_, FLContext
from mindspore_federated import log as logger

from ..startup.ssl_config import init_ssl_config
from ..startup.yaml_config import load_yaml_config
from ..common import _fl_context, check_type

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


class BroadcastNet(Cell):
    """
    Construct of weight input for Broadcast.
    """

    def __init__(self):
        super().__init__()
        self._broadcast = ops.Broadcast(0)

    def construct(self, input_x):
        return self._broadcast((input_x,))


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


def _get_lr(network):
    for sub_cell in network.cells():
        if isinstance(sub_cell, nn.Optimizer):
            return sub_cell.get_lr().asnumpy()
        lr = _get_lr(sub_cell)
        if lr is not None:
            return lr
    return None


class FederatedLearningManager(Callback):
    """
    Manage Federated Learning during training.

    Args:
        yaml_config (str): The yaml file path. For more detail see `federated_server_yaml <https://gitee.com/mindspore/federated/blob/master/docs/api/api_python_en/horizontal/federated_server_yaml.md>`_.
        model (nn.Cell): A model for Federated Training.
        sync_frequency (int): Synchronization frequency of parameters in Federated Learning. Indicating the number
                              of steps between two adjacent synchronization operations when `dataset_sink_mode` is
                              set to False. If `sync_type` is set to "fixed", it serves as a fixed number of steps.
                              If `sync_type` is set to "adaptive", it serves as the initial value of the adaptive
                              synchronization frequency. Note that its function is changed in dataset sink mode.
                              If `dataset_sink_mode` is set to True and `sink_size` is set to a non-positive value,
                              the synchronization operation will execute once every `sync_frequency` epochs. If
                              `dataset_sink_mode` is set to True and `sink_size` is set to a positive value, the
                              synchronization operation will execute once every `sink_size` * `sync_frequency` steps.
                              The `dataset_sink_mode` and the `sink_size` is set by users in `mindspore.train.Model` .
        http_server_address (str): The http server address used for communicating. Default: "".
        data_size (int): The data size to be reported to the worker. Default: 1.
        sync_type (str): The synchronization type of parameter in Federated Learning.
                         Supports ["fixed", "adaptive"]. Default: "fixed".

                         - fixed: The frequency of parameter synchronization is fixed.

                         - adaptive: The frequency of parameter synchronization changes adaptively.

        run_distribute (bool): Whether to open distribute training. Default: False.
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
                 run_distribute=False, ssl_config=None, **kwargs):
        super(FederatedLearningManager, self).__init__()
        check_type.check_str("yaml_config", yaml_config)
        init_ssl_config(ssl_config)
        load_yaml_config(yaml_config, _fl_context.ROLE_OF_SERVER)

        ctx = FLContext.get_instance()
        server_mode = ctx.server_mode()
        aggregation_type = ctx.aggregation_type()
        encrypt_type = ctx.encrypt_type()
        ctx.set_http_server_address(http_server_address)

        initial_model = {}
        for param in model.trainable_params():
            param_data = np.reshape(param.asnumpy(), -1)
            initial_model[param.name] = param_data
        Federated_.init_federated_worker(initial_model)

        validator.check_isinstance('model', model, nn.Cell)
        validator.check_positive_int(sync_frequency)
        validator.check_string(sync_type, ["fixed", "adaptive"])
        self._server_mode = server_mode
        self._model = model
        self._sync_frequency = sync_frequency
        self._next_begin_sync_iter = 1
        self._next_end_sync_iter = self._sync_frequency
        self._data_size = data_size
        self._sync_type = sync_type
        self._run_distribute = run_distribute
        if self._run_distribute:
            init()
            self._broadcast = BroadcastNet()
            self._rank_id = get_rank()
            logger.info(f"Rank id is {self._rank_id}")
        self._global_step = 0
        self._aggregation_type = aggregation_type
        self._global_prefix = "global_weights"
        if self._aggregation_type not in _fl_context.SUPPORT_AGG_TYPES and \
                self._server_mode == _fl_context.SERVER_MODE_CLOUD:
            raise ValueError(
                "aggregation_type must be in {}, but got {}.".format(_fl_context.SUPPORT_AGG_TYPES,
                                                                     self._aggregation_type))

        if self._aggregation_type in (_fl_context.FEDPROX, _fl_context.FEDNOVA):
            self._global_weights = ParameterTuple(self._model.trainable_params()).clone(prefix=self._global_prefix)
            for param in self._global_weights:
                param.requires_grad = False
                self._model.insert_param_to_cell(param.name, param, False)

        self._encrypt_type = encrypt_type
        if self._encrypt_type not in _fl_context.SUPPORT_ENC_TYPES_CLOUD and \
                self._server_mode == _fl_context.SERVER_MODE_CLOUD:
            raise ValueError(
                "encrypt_mode must be in {}, but got {}.".format(_fl_context.SUPPORT_ENC_TYPES_CLOUD,
                                                                 self._encrypt_type))
        if self._is_adaptive_sync():
            self._as_set_init_state(kwargs)
            self._as_wrap_cell()
        logger.info(f"Step number needs to run per iteration {self._next_end_sync_iter},"
                    f"server mode {self._server_mode}, aggregation type {self._aggregation_type},"
                    f"encrypt type {self._encrypt_type}, http server address {http_server_address}")
        self._fl_param_names = list()
        self._fl_param_names = _get_fl_param_names(self._model, self._fl_param_names)

        if not self._fl_param_names:
            self._fl_param_names = [_.name for _ in self._model.trainable_params()]
        self._last_params = dict()
        self._local_control_params = dict()
        self._global_control_params = dict()

        self._scaffold_prefix = "control."
        if self._is_scaffold():
            for param in self._model.trainable_params():
                if param.name in self._fl_param_names:
                    self._last_params[param.name] = deepcopy(param.asnumpy())
                    self._local_control_params[param.name] = np.zeros_like(param.asnumpy())
                    self._global_control_params[param.name] = np.zeros_like(param.asnumpy())

    def __del__(self):
        Federated_.stop_federated_worker()

    def _is_adaptive_sync(self):
        """
        Determine whether adaptive frequency synchronization is required.
        """
        return self._sync_type == "adaptive"

    def _is_scaffold(self):
        """
        Determine whether scaffold is required.
        """
        return self._aggregation_type == _fl_context.SCAFFOLD

    def _is_fednova(self):
        """
        Determine whether FedNova is required.
        """
        return self._aggregation_type == _fl_context.FEDNOVA

    def _as_set_init_state(self, kwargs):
        """
        Setting the initial state for adaptive synchronization.
        """
        self._as_prefix = "as_abs_grad."

        self._min_consistent_rate = kwargs.get("min_consistent_rate", 1.1)
        validator.check_non_negative_float(self._min_consistent_rate)
        self._min_consistent_rate_at_round = kwargs.get("min_consistent_rate_at_round", 0)
        validator.check_non_negative_int(self._min_consistent_rate_at_round)
        self._ema_alpha = kwargs.get("ema_alpha", 0.5)
        validator.check_float_range(self._ema_alpha, 0.0, 1.0, validator.INC_NEITHER)
        self._observation_window_size = kwargs.get("observation_window_size", 5)
        validator.check_positive_int(self._observation_window_size)
        self._frequency_increase_ratio = kwargs.get("frequency_increase_ratio", 2)
        validator.check_positive_int(self._frequency_increase_ratio)
        self._unchanged_round = kwargs.get("unchanged_round", 0)
        validator.check_non_negative_int(self._unchanged_round)

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

    def _update_model_with_distribute(self, weights, weight_infos):
        """
        Update model with distributed training mode.
        """
        if self._rank_id == 0:
            update_and_get_model = _UpdateAndGetModel(weights)
            feature_map = update_and_get_model.construct()
            if not feature_map:
                raise ValueError("Feature map from getting model is empty!")

            parameter_dict = {}
            for key, weight_info in weight_infos.items():
                if not feature_map[key]:
                    continue
                value = feature_map[key]
                shape, dtype = weight_info[0], weight_info[1]
                param_data = np.reshape(value, shape).astype(dtype)
                tensor = Tensor(param_data)
                parameter_dict[key] = Parameter(tensor, name=key)
                self._broadcast(tensor)
            load_param_into_net(self._model, parameter_dict)
        else:
            parameter_dict = {}
            for key, weight_info in weight_infos.items():
                value = weights[key]
                shape, dtype = weight_info[0], weight_info[1]
                param_data = np.reshape(value, shape).astype(dtype)
                received_tensor = self._broadcast(Tensor(param_data))
                parameter_dict[key] = Parameter(received_tensor[0], name=key)
            load_param_into_net(self._model, parameter_dict)

    def _update_model(self, weights, weight_infos):
        """
        Update and get model without distributed training mode.
        """
        update_and_get_model = _UpdateAndGetModel(weights)
        feature_map = update_and_get_model.construct()
        if not feature_map:
            raise ValueError("Feature map from getting model is empty!")

        parameter_dict = {}
        parameter_dict_global = {}
        for key, value in feature_map.items():
            if key not in weight_infos:
                continue
            shape, dtype = weight_infos[key]
            param_data = np.reshape(value, shape).astype(dtype)
            parameter_dict[key] = Parameter(Tensor(param_data), name=key)
            parameter_dict_global[self._global_prefix + "." + key] = \
                Parameter(Tensor(param_data), name=self._global_prefix + "." + key)
        load_param_into_net(self._model, parameter_dict)
        if self._aggregation_type in (_fl_context.FEDPROX, _fl_context.FEDNOVA):
            load_param_into_net(self._model, parameter_dict_global)

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

    def _scaffold_set_global_control_params(self, flattened_control_params):
        for name in self._global_control_params:
            control_name = self._scaffold_prefix + name
            if control_name in flattened_control_params:
                global_control_param = deepcopy(flattened_control_params[control_name])
                shape = self._global_control_params[name].shape
                self._global_control_params[name] = np.array(global_control_param, dtype=np.float32).reshape(shape)
            else:
                raise ValueError("'{}' is not in control parameters sent by server".format(control_name))

    def _scaffold_update_params(self, lr):
        """
        Using control parameters to update parameters every step.
        """
        for param in self._model.trainable_params():
            name = param.name
            if name in self._fl_param_names:
                if name in self._global_control_params:
                    global_control_param = self._global_control_params[name]
                else:
                    raise ValueError("'{}' is not in global_control_params".format(name))
                if name in self._local_control_params:
                    local_control_param = self._local_control_params[name]
                else:
                    raise ValueError("'{}' is not in local_control_params".format(name))
                control_params = lr * (global_control_param - local_control_param)
                param.set_data(Tensor(param.asnumpy() - control_params))

    def _scaffold_get_control_params(self, lr):
        """
        Get updated control parameters.
        """
        control_params = dict()
        for param in self._model.trainable_params():
            name = param.name
            if name in self._fl_param_names:
                if name in self._local_control_params:
                    local_control_param = deepcopy(self._local_control_params[name])
                else:
                    raise ValueError("'{}' is not in local_control_params".format(name))
                if name in self._global_control_params:
                    global_control_param = deepcopy(self._global_control_params[name])
                else:
                    raise ValueError("'{}' is not in global_control_params".format(name))
                temp1 = local_control_param - global_control_param
                if name in self._last_params:
                    temp2 = (self._last_params[name] - param.asnumpy()) / (self._sync_frequency * lr)
                else:
                    raise ValueError("'{}' is not in last_params".format(name))
                control_params[name] = temp1 + temp2
        return control_params

    def _scaffold_set_last_params_and_local_control_params(self, control_params):
        for param in self._model.trainable_params():
            name = param.name
            if name in self._fl_param_names:
                self._last_params[name] = deepcopy(param.asnumpy())
                if name in control_params:
                    self._local_control_params[name] = control_params[name]
                else:
                    raise ValueError("'{}' is not in control_params".format(name))

    def _model_params_to_weights_dict(self, weights, weight_infos):
        """Exact trainable params from model, then fill into weights and weights_infos"""
        for param in self._model.trainable_params():
            if self._global_prefix not in param.name:
                param_np = param.asnumpy()
                if param_np.dtype != np.float32:
                    continue
                weight_infos[param.name] = (param_np.shape, param_np.dtype)
                weights[param.name] = param_np.reshape(-1).tolist()

    def _model_params_to_weights_diff_dict(self, weights, weight_infos):
        """Exact trainable params from model, then calculate diff value for FedNova"""
        local_params = list(filter(lambda x: self._global_prefix not in x.name and x.requires_grad,
                                   self._model.get_parameters()))
        global_params = list(filter(lambda x: self._global_prefix in x.name and not x.requires_grad,
                                    self._model.get_parameters()))
        for local_param, global_param in zip(local_params, global_params):
            param = local_param - global_param
            param_np = param.asnumpy()
            weight_infos[local_param.name] = (param_np.shape, param_np.dtype)
            weights[local_param.name] = param_np.reshape(-1).tolist()

    def on_train_step_begin(self, run_context):
        self._global_step += 1
        is_cloud = self._server_mode == _fl_context.SERVER_MODE_CLOUD
        is_sync = self._global_step == self._next_begin_sync_iter
        is_dist = self._rank_id == 0 if self._run_distribute else not self._run_distribute

        if is_cloud and is_sync and is_dist:
            # In FedNova mode, the upload _data_size will be reset to the number of training steps
            if self._is_fednova():
                cb_params = run_context.original_args()
                self._data_size = cb_params.batch_num * self._sync_frequency \
                        if cb_params.dataset_sink_mode else self._sync_frequency
            start_fl_job = _StartFLJob(self._data_size)
            flattened_control_params = start_fl_job.construct()
            if self._is_scaffold() and self._global_step != 1:
                self._scaffold_set_global_control_params(flattened_control_params)
        logger.debug("run_context is %r", run_context)

    def on_train_step_end(self, run_context):
        lr = 0.0
        if self._is_scaffold():
            cb_params = run_context.original_args()
            train_network = cb_params.train_network
            lr = _get_lr(train_network)
            if lr is None:
                raise ValueError("Can not find optimizer in train network!")
            self._scaffold_update_params(lr)
        if self._server_mode == _fl_context.SERVER_MODE_CLOUD:
            if self._global_step == self._next_end_sync_iter:
                if self._is_adaptive_sync():
                    self._as_set_grads()
                if self._encrypt_type == _fl_context.ENCRYPT_STABLE_PW:
                    exchange_keys = _ExchangeKeys()
                    exchange_keys.construct()
                    get_keys = _GetKeys()
                    get_keys.construct()

                control_params = dict()
                if self._is_scaffold():
                    control_params = self._scaffold_get_control_params(lr)

                weights = {}
                weight_infos = {}
                if self._is_fednova():
                    self._model_params_to_weights_diff_dict(weights, weight_infos)
                else:
                    self._model_params_to_weights_dict(weights, weight_infos)

                if self._is_scaffold():
                    for name in control_params:
                        delta_control_param = control_params[name] - self._local_control_params[name]
                        weights[self._scaffold_prefix + name] = delta_control_param.reshape(-1).tolist()
                if self._run_distribute:
                    self._update_model_with_distribute(weights, weight_infos)
                else:
                    self._update_model(weights, weight_infos)
                if self._is_scaffold():
                    self._scaffold_set_last_params_and_local_control_params(control_params)
                logger.info("Load params from getting model into net, global step is {}.".format(self._global_step))
                self._next_end_sync_iter = self._global_step + self._sync_frequency
                self._next_begin_sync_iter = self._global_step + 1
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
