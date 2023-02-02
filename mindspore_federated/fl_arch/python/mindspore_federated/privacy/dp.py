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

import logging
import numpy as np
from mindspore import Tensor, ops


def _check_eps(eps):
    if not isinstance(eps, (int, float)):
        raise TypeError(f"Parameter eps must be an int or a float number, but {type(eps)} found.")
    if eps < 0:
        raise ValueError(f"Parameter eps cannot be negative, but got {eps}.")
    if eps > 100:
        logging.info('Parameter eps %f is far too large which may cause overflow and is reassigned to 100.', eps)
        eps = 100
    return float(eps)


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
        self.eps = _check_eps(eps)
        logging.info('The label is protected by LabelDP with eps %f.', self.eps)
        self._onehot = ops.OneHot()

    def __call__(self, label):
        """
        input a label batch, output a perturbed label batch satisfying label differential privacy.
        """
        if not isinstance(label, Tensor):
            raise TypeError(f"The label must be a Tensor, but got {type(label)}")

        ones_cnt = np.sum(label.asnumpy() == 1)
        zeros_cnt = np.sum(label.asnumpy() == 0)
        if ones_cnt + zeros_cnt != label.size:
            raise ValueError(f"Invalid label form: the elements should be either 0 or 1.")

        if label.ndim == 1 or (label.ndim == 2 and label.shape[1] == 1):
            flip_prob = 1 / (np.exp(self.eps) + 1)
            binomial = np.random.binomial(1, flip_prob, label.shape)
            dp_label = (label - Tensor(binomial, dtype=label.dtype)).abs()
        elif label.ndim == 2:
            if ones_cnt != len(label):
                raise ValueError('Invalid one-hot form: each label should contain only a single 1.')
            keep_prob = np.exp(self.eps) / (np.exp(self.eps) + label.shape[1] - 1)
            flip_prob = 1 / (np.exp(self.eps) + label.shape[1] - 1)
            prob_array = label * (keep_prob - flip_prob) + Tensor(np.ones(label.shape)) * flip_prob
            dp_index = np.array([np.random.choice(label.shape[1], p=prob/sum(prob)) for prob in prob_array])
            dp_label = Tensor(np.eye(label.shape[1])[dp_index], dtype=label.dtype)
        else:
            raise ValueError(f"Invalid label dim: the dim must be 1 or 2.")

        return dp_label


class EmbeddingDP:
    """
    This class uses unary quantization and random response technique to create differentially private embeddings.

    Args:
        eps (Union[None, int, float]): the privacy budget which controls the level of differential privacy.

    Inputs:
        - **embedding** (Tensor) - a batch of embeddings that need protection.

    Outputs:
        Tensor, has the same shape and data type as `embedding`.

    Raises:
        TypeError: If `eps` is not a float or int.
        TypeError: If `embedding` is not a Tensor.
        ValueError: If `eps` is less than zero.

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore_federated.privacy import EmbeddingDP
        >>> ori_tensor = Tensor([1.5, -0.6, 7, -10])
        >>> dp_tensor = EmbeddingDP()(ori_tensor)
        >>> print(dp_tensor)
        [1., 0., 1., 0.]
    """
    def __init__(self, eps=None) -> None:
        self.eps = eps
        if eps or eps == 0:
            self.eps = _check_eps(eps)
            self.q = 1 / (ops.exp(Tensor(self.eps / 2)) + 1)
            self.p = 1 - self.q
            logging.info('The embedding is protected by EmbeddingDP with eps %f.', self.eps)
        else:
            logging.info('Parameter eps is missing and the embedding is protected with quantization only.')

    def __call__(self, embedding):
        if not isinstance(embedding, Tensor):
            raise TypeError(f'The embedding must be a Tensor, but got {type(embedding)}.')
        embedding = self._unary_encoding(embedding)
        if self.eps or self.eps == 0:
            embedding = self._randomize(embedding)
        return embedding

    def _unary_encoding(self, embedding):
        embedding[embedding > 0] = 1
        embedding[embedding <= 0] = 0
        return embedding

    def _randomize(self, embedding):
        p_binomial = np.random.binomial(1, self.p, embedding.shape)
        q_binomial = np.random.binomial(1, self.q, embedding.shape)
        return Tensor(np.where(embedding.asnumpy() == 1, p_binomial, q_binomial), dtype=embedding.dtype)
