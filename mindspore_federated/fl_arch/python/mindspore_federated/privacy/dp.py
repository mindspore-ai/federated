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
"""classes of differential privacy mechanisms for the federated learner."""

import numpy as np
import mindspore as ms
from mindspore import Tensor, ops
from mindspore import numpy as mnp

class LabelDP:
    """
    Label differential privacy module.

    This class uses the Random Response algorithm to create differentially private label based on the input label.
    Currently only support binary labels (dim = 1 or 2) and onehot labels (dim = 2).

    Args:
        eps (Union[int, float]): the privacy parameter, representing the level of differential privacy protection.

    Inputs:
        - **label** (Tensor) - a batch of labels to be made differentially private.

    Outputs:
        Tensor, has the same shape and data type as `label`.

    Raises:
        TypeError: If `eps` is not a float or int.
        TypeError: If `label` is not a Tensor.
        ValueError: If `eps` is less than zero.

    Examples:
        >>> # make private a batch of binary labels
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore_federated.privacy import LabelDP
        >>> label_dp = LabelDP(eps=0.0)
        >>> label = Tensor(np.zeros((5, 1)), dtype=mindspore.float32)
        >>> dp_label = label_dp(label)
        >>> print(dp_label)
        [[1.]
         [0.]
         [0.]
         [1.]
         [1.]]
        ...
        >>> # make private a batch of onehot labels
        >>> label = Tensor(np.hstack((np.ones((5, 1)), np.zeros((5, 2)))), dtype=mindspore.float32)
        >>> dp_label = label_dp(label)
        >>> print(dp_label)
        [[1. 0. 0.]
         [0. 1. 0.]
         [1. 0. 0.]
         [0. 0. 1.]
         [0. 1. 0.]]
    """

    def __init__(self, eps) -> None:
        self.eps = eps

        if not isinstance(self.eps, int) and not isinstance(self.eps, float):
            raise TypeError('LabelDP: eps must be an int or a float, but found {}'.format(type(self.eps)))
        self.eps = float(self.eps)
        if self.eps < 0:
            raise ValueError('LabelDP: eps must be greater than or equal to zero, but got {}'.format(self.eps))

        self._onehot = ops.OneHot()

    def __call__(self, label):
        """
        input a label batch, output a perturbed label batch satisfying label differential privacy.
        """
        if not isinstance(label, Tensor):
            raise TypeError('LabelDP: the input label must be a Tensor, but got {}'.format(type(label)))

        if (len(label.shape) == 2 and label.shape[1] == 1) or len(label.shape) == 1:
            # binary label
            if sum(label == 1) + sum(label == 0) != label.shape[0]:
                raise ValueError('Unsupported labels. LabelDP currently only support binary or onehot labels.')
            flip_prob = 1 / (1 + np.exp(self.eps))
            flip = Tensor(np.random.binomial(1, flip_prob, size=label.shape), dtype=label.dtype)
            dp_label = mnp.abs(flip - label)
        elif len(label.shape) == 2:
            # onehot label
            if mnp.sum(label == 1) != label.shape[0] or mnp.sum(label == 1) + mnp.sum(label == 0) != label.size:
                raise ValueError('Unsupported labels. LabelDP currently only support binary or onehot labels.')
            dp_label = label * self.eps + Tensor(np.random.laplace(0, 1, label.shape), ms.float32)
            dp_label = ops.Argmax(output_type=ms.int32)(dp_label)
            dp_label = self._onehot(dp_label, label.shape[1], Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
            if dp_label.dtype != label.dtype:
                dp_label = Tensor(dp_label, dtype=label.dtype)
        else:
            raise ValueError(f'''LabelDP currently only support binary or onehot labels, so the dim of the input labels
                             is expected to be 1 or 2, but got {len(label.shape)}''')
        return dp_label
