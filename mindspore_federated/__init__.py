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


def _mindspore_version_check():
    """
    Do the MindSpore version check for MindSpore Reinforcement. If the
    MindSpore can not be imported, it will raise ImportError. If its
    version is not compatibale with current MindSpore Reinforcement verision,
    it will print a warning.

    Raise:
        ImportError: If the MindSpore can not be imported.
    """

    try:
        import mindspore as ms
        from mindspore import log as logger
    except (ImportError, ModuleNotFoundError):
        print("Can not find MindSpore in current environment. Please install "
              "MindSpore before using MindSpore Reinforcement, by following "
              "the instruction at https://www.mindspore.cn/install")
        raise

    ms_msrl_version_match = {'0.1.0': ['1.7.0', '1.8.0', '1.9.0']}

    ms_version = ms.__version__
    required_mindspore_verision = ms_msrl_version_match[version]

    if ms_version not in required_mindspore_verision:
        logger.warning("Current version of MindSpore is not compatible with MindSpore Federated. "
                       "Some functions might not work or even raise error. Please install MindSpore "
                       "version == {}. For more details about dependency setting, please check "
                       "the instructions at MindSpore official website https://www.mindspore.cn/install "
                       "or check the README.md at https://gitee.com/mindspore/federated"
                       .format(required_mindspore_verision))
        warning_countdown = 3
        for i in range(warning_countdown, 0, -1):
            logger.warning(
                f"Please pay attention to the above warning, countdonw: {i}")
            time.sleep(1)


_mindspore_version_check()
