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
import os

import pandas as pd
from .base_data import BaseData


class PandasData(BaseData):
    """
    Pandas Data format.
    """
    # using this field to do register
    reg_key = "csv"

    def __init__(self, store_type=None, primary_key=None, schema=None, desc=None,
                 config: dict = None):
        super().__init__()
        self._store_type = store_type
        if not self._store_type == "csv":
            raise ValueError("store type: {} is not support currently".format(self._store_type))
        if config['main_table_files'] is None:
            raise ValueError("The main table files are None.")
        # The raw data paths, which must be set in both leader and follower, type is (Union(list(str), str):.
        self._main_table_files = config['main_table_files']
        self._store = pd.DataFrame()
        self._primary_key = primary_key
        self._schema = dict() if schema is None else schema
        self._pd_schema = dict() if schema is None else {_: schema[_]["type"] for _ in schema}
        self._usecols = None if schema is None else [_ for _ in schema]
        self._desc = desc

    def read_csv_data(self, data_path):
        df = pd.read_csv(data_path, index_col=self._primary_key, usecols=self._usecols, dtype=self._pd_schema)
        df.index = df.index.fillna("")
        self.merge(df)

    def load_raw_data(self):
        """
        Load data from the file system. Only support "csv, mysql" currently.
        """
        main_table_files = self._main_table_files
        if isinstance(main_table_files, list):
            for main_table_file in main_table_files:
                self.read_csv_data(main_table_file)
        elif isinstance(main_table_files, str):
            if os.path.isdir(main_table_files):
                index = 0
                for main_table_file in os.listdir(main_table_files):
                    index += 1
                    main_table_file = os.path.join(main_table_files, main_table_file)
                    self.read_csv_data(main_table_file)
            else:
                self.read_csv_data(main_table_files)
        else:
            raise TypeError("main_table_files must be list or str, but get {}".format(
                type(main_table_files)))


    def keys(self):
        return self._store.index

    def values(self, keys=None):
        if keys is None:
            df = self._store
        else:
            df = self._store.loc[keys]
        column_names = df.columns.tolist()
        column_names.insert(0, self._primary_key)
        for value in df.itertuples():
            feature = dict()
            for single_value, column_name in zip(value, column_names):
                if column_name not in self._pd_schema:
                    continue
                if pd.isna(single_value):
                    if self._pd_schema[column_name] != "string":
                        raise ValueError("The column: '{}' has a null number.".format(column_name))
                    single_value = ""
                if self._pd_schema[column_name] == "bytes":
                    single_value = bytes(single_value, encoding="utf-8")
                feature[column_name] = single_value
            yield feature

    def merge(self, store):
        self._store = pd.concat([self._store, store])
        if not self._store.index.is_unique:
            duplicate_num = self._store.index.duplicated().sum()
            raise ValueError("There are {} duplicated ids".format(duplicate_num))

    def verify(self):
        """
        Verify main_table_files.
        """
        self._verify_schema()
        main_table_files = self._main_table_files
        if isinstance(main_table_files, list):
            for main_table_file in main_table_files:
                if not os.path.isfile(main_table_file):
                    raise ValueError("{} in main_table_files is not a file.".format(main_table_file))
        elif isinstance(main_table_files, str):
            if not os.path.exists(main_table_files):
                raise ValueError("main_table_files: {} is not exist.".format(main_table_files))
            if os.path.isdir(main_table_files):
                for main_table_file in os.listdir(main_table_files):
                    main_table_file = os.path.join(main_table_files, main_table_file)
                    if not os.path.isfile(main_table_file):
                        raise ValueError("{} in main_table_files is not a file.".format(main_table_file))
        else:
            raise TypeError("main_table_files must be list or str, but get {}".format(type(main_table_files)))
