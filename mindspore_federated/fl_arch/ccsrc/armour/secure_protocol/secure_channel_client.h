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

#ifndef SECURE_CHANNEL_CLIENT_H_INCLUDED
#define SECURE_CHANNEL_CLIENT_H_INCLUDED

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <secGear/enclave.h>
#include "armour/secure_protocol/secure_channel.h"

#ifdef  __cplusplus
extern "C" {
#endif

int get_client_ecdhkey(key_exchange_t *key_exchange_info, const uint8_t *rsa_pubkey, size_t rsa_pubkey_len,
    uint8_t *enclave_key, size_t enclave_key_len, const uint8_t *client_sign_key, size_t client_sign_key_len,
    uint8_t **client_key_p, size_t *client_key_len);

int secure_channel_read(key_exchange_t *key_exchange_info, bool is_server, uint8_t *data, size_t data_len, uint8_t *out,
    size_t out_buf_len);

int secure_channel_write(key_exchange_t *key_exchange_info, bool is_server, uint8_t *plain, size_t plain_len,
    uint8_t *out, size_t out_buf_len);

# ifdef  __cplusplus
}
# endif
#endif
