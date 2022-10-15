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
"""Store by pandas."""

import pandas as pd
from .base_data import BaseData


class PandasData(BaseData):
    """
    Pandas Data format.
    """

    def __init__(self, store=None, primary_key=None, schema=None, desc=None):
        super().__init__()
        self._store = pd.DataFrame() if store is None else store
        self._primary_key = primary_key
        self._schema = dict() if schema is None else schema
        self._desc = desc
        self._pd_schema = dict() if schema is None else {_: schema[_]["type"] for _ in schema}
        self._usecols = None if schema is None else [_ for _ in schema]
        self._need_sorted = True

    def load_raw_data(self, data_path):
        df = pd.read_csv(data_path, usecols=self._usecols, dtype=self._pd_schema)
        self.merge(df)

    def keys(self):
        if self._primary_key is None:
            return list(self._store.iloc[:, 0])
        return list(self._store.loc[:, self._primary_key])

    def values(self, keys=None):
        if self._need_sorted:
            self._store = self._store.sort_values(by=self._primary_key, ascending=False)
            self._need_sorted = False
        if keys is None:
            values = self._store.values
        else:
            if self._primary_key is None:
                values = self._store.iloc[:, 0]
            else:
                values = self._store.loc[:, self._primary_key]
            values = self._store[values.isin(keys)].values
        feature_names = list(_ for _ in self._store)
        for value in values:
            feature = dict()
            for single_value, feature_name in zip(value, feature_names):
                feature[feature_name] = single_value
            yield feature

    def merge(self, store):
        origin_len = len(self._store) + len(store)
        self._store = pd.concat([self._store, store])
        self._store.drop_duplicates(subset=[self._primary_key], inplace=True)
        drop_duplicate_len = len(self._store)
        if drop_duplicate_len != origin_len:
            raise ValueError("There are {} duplicated ids".format(origin_len - drop_duplicate_len))
