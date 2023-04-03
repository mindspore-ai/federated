# Copyright 2023 Huawei Technologies Co., Ltd
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
"""base data processing"""

from mindspore_federated.common import _checkparam as validator
from mindspore_federated.common.check_type import check_str
from mindspore_federated.data_join.store.data_source_mgr import DataSourceMete

SUPPORT_TYPES = ("int32", "int64", "float32", "float64", "string", "bytes")
SUPPORT_ARRAY_TYPES = ("int32", "int64", "float32", "float64")


class BaseData(metaclass=DataSourceMete):
    """Abstract base data source"""

    def __init__(self, schema=None, desc=None):
        super().__init__()
        self._schema = dict() if schema is None else schema
        self._desc = desc

    def keys(self):
        pass

    def values(self, keys=None):
        pass

    def schema(self):
        return self._schema

    def desc(self):
        return self._desc

    def verify(self):
        pass

    def _verify_schema(self):
        """
        Verify schema.
        """
        if isinstance(self._schema, dict):
            for key in self._schema:
                check_str(arg_name="column name", str_val=key)

                shape = self._schema[key].get("shape")
                data_type = self._schema[key].get("type")

                if shape is not None:
                    if isinstance(shape, list):
                        raise TypeError("shape must be list, but get {}".format(type(shape)))
                else:
                    shape = (1,)

                if data_type is not None:
                    if len(shape) == 1:
                        validator.check_string(data_type, SUPPORT_TYPES, arg_name="data type")
                    else:
                        validator.check_string(data_type, SUPPORT_ARRAY_TYPES, arg_name="array data type")
        else:
            raise TypeError("schema must be dict, but get {}".format(type(self._schema)))
