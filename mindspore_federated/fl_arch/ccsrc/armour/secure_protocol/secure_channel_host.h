/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * CloudEnclave is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 * http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
 * PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef SECURE_CHANNEL_HOST_H_INCLUDED
#define SECURE_CHANNEL_HOST_H_INCLUDED

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <secGear/enclave.h>
#include "armour/secure_protocol/secure_channel.h"

#ifdef  __cplusplus
extern "C" {
#endif

int save_enclave_rsakey_to_file(cc_enclave_t *context, size_t key_id, const char *file_name);

int set_enclave_rsakey_from_file(cc_enclave_t *context, size_t key_id, const char *file_name);

# ifdef  __cplusplus
}
# endif
#endif
