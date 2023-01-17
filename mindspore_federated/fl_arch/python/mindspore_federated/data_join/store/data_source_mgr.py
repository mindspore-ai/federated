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
"""data source mgr to create data source"""


class DataSourceMgr:
    """Data source mgr"""
    data_source_map = dict()

    @classmethod
    def register(cls, name, obj):
        cls.data_source_map[name] = obj

    @classmethod
    def get_data_source_cls(cls, name):
        return cls.data_source_map.get(name)


class DataSourceMete(type):
    """Mete class of data source"""

    def __new__(mcs, cls_name, cls_base, attrs):
        cls_obj = super(DataSourceMete, mcs).__new__(mcs, cls_name, cls_base, attrs)
        if cls_name != "BaseData":
            DataSourceMgr.register(cls_obj.reg_key, cls_obj)
        return cls_obj
