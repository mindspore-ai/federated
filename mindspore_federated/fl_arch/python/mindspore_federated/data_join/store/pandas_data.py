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

    def load_raw_data(self, data_path):
        df = pd.read_csv(data_path, index_col=self._primary_key, usecols=self._usecols, dtype=self._pd_schema)
        self.merge(df)

    def keys(self):
        return self._store.index

    def values(self, keys=None):
        if keys is None:
            values = self._store.values
        else:
            values = self._store.loc[keys].values
        for key, value in zip(keys, values):
            feature = {self._primary_key: key}
            for single_value, column_name in zip(value, self._store):
                if self._pd_schema[column_name] != "string":
                    if pd.isna(single_value):
                        raise ValueError("The column: [{}] and the key: [{}] is null.".format(column_name, key))
                else:
                    single_value = ""
                feature[column_name] = single_value.astype(self._pd_schema[column_name])
            yield feature

    def merge(self, store):
        self._store = pd.concat([self._store, store])
        if not self._store.index.is_unique:
            duplicate_num = self._store.index.duplicated().sum()
            raise ValueError("There are {} duplicated ids".format(duplicate_num))
