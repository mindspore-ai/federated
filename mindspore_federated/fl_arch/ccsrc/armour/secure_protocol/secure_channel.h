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

#ifndef SECURE_CHANNEL_ENCLAVE_H
#define SECURE_CHANNEL_ENCLAVE_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <openssl/rsa.h>
#include <openssl/evp.h>

#ifdef  __cplusplus
extern "C" {
#endif

#define KEY_ID_LEN 16
#define RANDOM_LEN 32
#define KEYEX_CURE_ID_LEN 2
#define KEYEX_PUBKEY_LEN 1
#define KEYEX_SIG_LEN 2
#define MIN_ENCLAVE_KEYEX_LEN 37
#define DATA_SIZE_LEN 2
#define DATA_SIZE_NUM 2
#define MAX_DATA_LEN 0xffff
#define MAX_KEY_LEN 64
#define SECURE_KEY_LEN 32
#define SECURE_IV_LEN 16
#define GCM_TAG_LEN 16
#define SEQ_MAX_LEN 8
#define ADD_DATA_TEXT "seal rsa key"
#define MAX_SESSION_NUM 1031
#define INVALID_SESSION_ID ((size_t)(-1))
#define KEY_LEN_128 (128 / 8)
#define KEY_LEN_192 (192 / 8)
#define KEY_LEN_256 (256 / 8)
#define AES_BATCH_LEN 128
#define ECC_POINT_COMPRESSED_MULTIPLY 2
#define MAX_ECC_PUBKEY_LEN 255
#define BYTE_TO_BIT_LEN 8
#define IS_EVEN_NUMBER 2
#define SECURE_CHANNEL_ERROR (-1)

/* count the current number of sessions, [0, MAX_SESSION_NUM] */
extern size_t g_cur_session_num;
typedef struct _key_exchange_t {
    size_t session_id;
    void *session_info;
    bool client_auth;
    int ec_nid;
    size_t pubkey_len;
    size_t key_len;
    EC_KEY *ecdh_key;
    size_t sig_len;
    RSA *signed_key;
    uint8_t client_random[RANDOM_LEN];
    uint8_t server_random[RANDOM_LEN];
    uint8_t shared_key[MAX_KEY_LEN];
    uint8_t client_write_key[SECURE_KEY_LEN];
    uint8_t server_write_key[SECURE_KEY_LEN];
    uint8_t client_write_iv[SECURE_IV_LEN];
    uint8_t server_write_iv[SECURE_IV_LEN];
    size_t client_write_seq;
    size_t server_write_seq;
} key_exchange_t;

typedef struct _aes_algorithm_param {
    uint8_t *plain;
    int plain_len;
    uint8_t *cipher;
    int cipher_len;
    uint8_t *aad;
    int aad_len;
    uint8_t *key;
    int key_len;
    uint8_t *iv;
    int iv_len;
    uint8_t *tag;
    int tag_len;
} aes_param_t;

# ifdef  __cplusplus
}
# endif
#endif
