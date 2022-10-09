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
"""Interface for set feature_map"""
import numpy as np


class FeatureItem:
    """
    Feature Item
    """
    def __init__(self, feature_name, data, require_aggr=True):
        if not isinstance(data, np.ndarray):
            raise RuntimeError(f"The type of parameter 'data' is expected to be instance of numpy.ndarray")
        if data.dtype != np.float32:
            raise RuntimeError(f"The value type of parameter 'data' is expected to be float32, but got {data.dtype}")
        if data.size == 0:
            raise RuntimeError(f"The value size of parameter 'data' is expected to > 0")
        if not data.flags['FORC']:
            data = np.ascontiguousarray(data)
        self.data_ = data
        self.feature_name_ = feature_name
        self.require_aggr_ = require_aggr

    @property
    def feature_name(self):
        return self.feature_name_

    @property
    def data(self):
        return self.data_

    @property
    def shape(self):
        return self.data_.shape

    @property
    def require_aggr(self):
        return self.require_aggr_

    @require_aggr.setter
    def require_aggr(self, require_aggr):
        self.require_aggr_ = require_aggr


class FeatureMap:
    """
    Feature Map
    """
    def __init__(self):
        self.feature_map_ = {}

    def add_feature(self, feature_name, tensor, require_aggr=True):
        feature = FeatureItem(feature_name, tensor, require_aggr)
        self.feature_map_[feature_name] = feature
        return feature

    def feature_map(self):
        return self.feature_map_
