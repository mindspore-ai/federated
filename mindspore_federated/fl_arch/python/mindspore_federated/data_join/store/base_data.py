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

class BaseData(object):
    def __init__(self, store=None, schema=None, desc=None):
        super().__init__()
        self._store = dict() if store is None else store
        self._it = iter(self._store)
        self._schema = dict() if schema is None else schema
        self._desc = desc

    def set(self, store):
        self._store = store

    def merge(self, store):
        origin_len = len(self._store) + len(store)
        self._store = dict(self._store, **store)
        drop_duplicate_len = len(self._store)
        if drop_duplicate_len != origin_len:
            raise ValueError("There are {} duplicated ids".format(origin_len - drop_duplicate_len))

    def keys(self):
        return list(self._store.keys())

    def values(self, keys=None):
        if keys is None:
            return list(self._store.values())
        return [self._store[_] for _ in keys]

    def schema(self):
        return self._schema

    def desc(self):
        return self._desc

    def size(self):
        return len(self._store)

    def __iter__(self):
        self._it = iter(self._store)
        return self

    def __next__(self):
        return next(self._it)
