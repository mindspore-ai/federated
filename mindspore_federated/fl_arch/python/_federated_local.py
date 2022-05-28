# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Interface for start up single core servable"""

from mindspore_federated.python.common import _fl_context
from mindspore_federated._mindspore_federated import Federated_

class FederatedLearningJob:
    r"""
    Servable startup configuration.

    For more detail, please refer to
    `MindSpore-based Inference Service Deployment <https://www.mindspore.cn/federated/docs/zh-CN/master/federated_example.html>`_ and
    `Servable Provided Through Model Configuration <https://www.mindspore.cn/federated/docs/zh-CN/master/federated_model.html>`_.

    Args:
        servable_directory (str): The directory where the servable is located in. There expects to has a directory
            named `servable_name`.
        servable_name (str): The servable name.
        device_ids (Union[int, list[int], tuple[int]], optional): The device list the model loads into and runs in.
            Used when device type is Nvidia GPU, Ascend 310/710/910. Default None.
        version_number (int, optional): Servable version number to be loaded. The version number should be a positive
            integer, starting from 1, and 0 means to load the latest version. Default: 0.
        device_type (str, optional): Currently supports "Ascend", "GPU", "CPU" and None. Default: None.

            - "Ascend": the platform expected to be Ascend 310/710/910, etc.
            - "GPU": the platform expected to be Nvidia GPU.
            - "CPU": the platform expected to be CPU.
            - None: the platform is determined by the MindSpore environment.

        num_parallel_workers (int, optional): This feature is currently in beta.
            The number of processes processing python tasks, at least the number
            of device cards used specified by the parameter device_ids. It will be adjusted to the number of device
            cards when it is less than the number of device cards. Default: 0.
        dec_key (bytes, optional): Byte type key used for decryption. The valid length is 16, 24, or 32. Default: None.
        dec_mode (str, optional): Specifies the decryption mode, take effect when dec_key is set.
            Option: 'AES-GCM' or 'AES-CBC'. Default: 'AES-GCM'.

    Raises:
        RuntimeError: The type or value of the parameters are invalid.
    """

    def __init__(self, kwargs):
        self.kwargs_ = kwargs

    def run(self):
        _fl_context._set_fl_context(self.kwargs_)
        return self.start_federated_job()

    def start_federated_job(self):
        return Federated_.start_federated_job()