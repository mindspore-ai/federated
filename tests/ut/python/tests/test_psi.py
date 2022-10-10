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
"""Test the functions of PSIDemo"""

from mindspore_federated._mindspore_federated import RunPSIDemo
from common import fl_test


def compute_right_result(alice_input, bob_input):
    alice_input_set = set(alice_input)
    bob_input_set = set(bob_input)
    return alice_input_set.intersection(bob_input_set)


def check_psi_is_ok(actual_result, psi_result):
    psi_set_result = set(psi_result)
    wrong_num = len(psi_set_result.difference(actual_result).union(
        actual_result.difference(psi_set_result)))
    return wrong_num


@fl_test
def test_case_from_eazy_list():
    """
    Feature: Test psi demo without communication
    Description: Input constructed through the easy list
    Expectation: success or ERROR with actual_result != psi_result.
    """
    server_input = ["1", "2", "3"]
    client_input = ["2", "3", "4", "5"]
    psi_result = RunPSIDemo(server_input, client_input, 0)
    actual_result = compute_right_result(server_input, client_input)
    assert not bool(check_psi_is_ok(actual_result, psi_result))
