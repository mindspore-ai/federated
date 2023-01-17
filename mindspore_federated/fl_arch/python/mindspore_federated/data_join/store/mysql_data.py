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
"""Store by mysql."""

import pymysql
from .base_data import BaseData


class MysqlData(BaseData):
    """
    Mysql Data format.
    """
    # using this field to do register
    reg_key = "mysql"

    def __init__(self, store_type=None, primary_key=None, schema=None, desc=None, config: dict = None):
        super().__init__()
        self._store_type = store_type
        if not self._store_type == "mysql":
            raise ValueError("store type: {} is not support currently".format(self._store_type))
        self._conn = pymysql.connect(host=config['mysql_host'],
                                     port=config['mysql_port'],
                                     database=config['mysql_database'],
                                     charset=config['mysql_charset'],
                                     user=config['mysql_user'],
                                     password=config['mysql_password'])
        self._cursor = self._conn.cursor()
        self._table_name = config['mysql_table_name']
        self._primary_key = primary_key
        self._schema = dict() if schema is None else schema
        self._desc = desc
        self._keys = []
        self._values = {}

    def load_raw_data(self):
        """Init raw data"""
        key_sql = "select {} from {}".format(self._primary_key, self._table_name)
        self._cursor.execute(key_sql)
        keys = self._cursor.fetchall()
        for key in keys:
            self._keys.append(key[0])

        feature_sql = "select "
        for key, _ in self._schema.items():
            feature_sql = feature_sql + key + ","
        feature_sql = feature_sql[0: len(feature_sql) - 1]
        feature_sql = feature_sql + " from {}".format(self._table_name)
        self._cursor.execute(feature_sql)
        values = self._cursor.fetchall()

        names = list(self._schema.keys())
        for value in values:
            value_dict = {}
            oaid = value[0]
            for i in range(len(value)):
                value_dict[names[i]] = value[i]
            self._values[oaid] = value_dict

    def keys(self):
        return self._keys

    def values(self, keys=None):
        values = []
        for key in keys:
            if key not in self._values:
                continue
            values.append(self._values[key])
        return values

    def __del__(self):
        self._cursor.close()
        self._conn.close()

    def verify(self):
        """
        Verify schema and connections.
        """
        self._verify_schema()
