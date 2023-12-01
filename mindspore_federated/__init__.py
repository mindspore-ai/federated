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
"""
Components for MindSpore Federated Learning Framework.
"""

import time
from mindspore_federated.version import __version__
from .fl_arch.python.mindspore_federated import *
from .fl_arch.python.mindspore_federated.trainer._fl_manager import FederatedLearningManager
from .fl_arch.python.mindspore_federated.startup.federated_local import FlSchedulerJob
from .fl_arch.python.mindspore_federated.startup.federated_local import FLServerJob

def _mindspore_version_check():
    """
    Do the MindSpore version check for MindSpore Federated. If the
    MindSpore can not be imported, it will raise ImportError. If its
    version is not compatible with current MindSpore Federated version,
    it will print a warning.

    Raise:
        ImportError: If the MindSpore can not be imported.
    """

    try:
        import mindspore as ms
        from mindspore import log as logger
    except (ImportError, ModuleNotFoundError):
        print("Can not find MindSpore in current environment. Please install "
              "MindSpore before using MindSpore Federated, by following "
              "the instruction at https://www.mindspore.cn/install")
        raise

    ms_fl_version_match = {'0.2.0': ['2.1']}

    ms_version = ms.__version__[:3]
    required_mindspore_version = ms_fl_version_match[__version__]

    if ms_version not in required_mindspore_version:
        logger.warning("Current version of MindSpore is not compatible with MindSpore Federated. "
                       "Some functions might not work or even raise error. Please install MindSpore "
                       "version == {}. For more details about dependency setting, please check "
                       "the instructions at MindSpore official website https://www.mindspore.cn/install "
                       "or check the README.md at https://gitee.com/mindspore/federated"
                       .format(required_mindspore_version))
        warning_countdown = 3
        for i in range(warning_countdown, 0, -1):
            logger.warning(
                f"Please pay attention to the above warning, countdonw: {i}")
            time.sleep(1)


_mindspore_version_check()

__all__ = [
    'FederatedLearningManager',
    'FlSchedulerJob',
    'FLServerJob',
    'VerticalFederatedCommunicator',
    'ServerConfig',
]
