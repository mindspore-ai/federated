/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * secGear is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *     http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
 * PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "armour/secure_protocol/tee_cutlayer.h"
#ifdef ENABLE_TEE
#include <secGear/enclave.h>
#include "armour/secure_protocol/enclave_call.h"
#include "armour/secure_protocol/secure_channel_host.h"
#include "armour/secure_protocol/secure_channel_client.h"
#endif

#ifdef ENABLE_TEE
/* OCall functions */
void ocall_print_string(const char *str) {
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    printf("%s", str);
}

void ocall_EIGEN_CPUID(int32_t *abcd, uint32_t func, uint32_t id) {
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    __asm__ __volatile__ ("xchgl %%ebx, %k1;cpuid; xchgl %%ebx,%k1":
                          "=a" (abcd[0]), "=&r" (abcd[1]), "=c" (abcd[2]), "=d" (abcd[3]) : "a" (func), "c" (id));
}
#endif

namespace mindspore {
namespace fl {
namespace TEE {
#ifdef ENABLE_TEE
cc_enclave_t *context = NULL;
size_t key_id = 0, session_id = 0;
uint8_t *rsa_pubkey = NULL, *enclave_key = NULL, *client_key = NULL;
key_exchange_t *ckey_exchange_info = NULL;

int establish_secure_channel() {
    printf("establish_secure_channel\n");
    size_t rsa_key_len = 3072, rsa_pub_exponent = RSA_F4;
    size_t sig_len = 0;
    size_t rsa_pubkey_len = 0, enclave_key_len = 0, client_key_len = 0;

    int  retval = 0;

    cc_enclave_result_t res = CC_FAIL;

    res = (cc_enclave_result_t)generate_enclave_rsakey(context, &retval, key_id, rsa_key_len, rsa_pub_exponent,
                                      NULL, 0, &rsa_pubkey_len, NULL, 0, &sig_len);
    if (res != CC_SUCCESS || retval != static_cast<int>(CC_ERROR_SHORT_BUFFER) || rsa_pubkey_len == 0) {
        printf("generate_enclave_rsakey error: %x!\n", res);
        return free_tee_cut_layer();
    }

    rsa_pubkey = reinterpret_cast<uint8_t *>(malloc(rsa_pubkey_len));
    if (rsa_pubkey == NULL) {
        printf("rsa_pubkey malloc error!\n");
        return free_tee_cut_layer();
    }
    res = (cc_enclave_result_t)get_enclave_rsa_pubkey(context, &retval, key_id, rsa_pubkey, rsa_pubkey_len,
                                     &rsa_pubkey_len, NULL, 0, &sig_len);
    if (res != CC_SUCCESS || retval != static_cast<int>(CC_SUCCESS)) {
        printf("get_enclave_rsa_pubkey error: %x!\n", res);
        return free_tee_cut_layer();
    }
    printf("get_enclave_rsa_pubkey success!\n");

    res = (cc_enclave_result_t)generate_enclave_ecdhkey(context, &retval, session_id, key_id,
                                       NID_brainpoolP256r1, NULL, 0, &enclave_key_len);
    if (res != CC_SUCCESS || retval != static_cast<int>(CC_ERROR_SHORT_BUFFER) || enclave_key_len == 0) {
        printf("generate_enclave_ecdhkey error: %x!\n", res);
        return free_tee_cut_layer();
    }

    enclave_key = reinterpret_cast<uint8_t *>(malloc(enclave_key_len));
    if (enclave_key == NULL) {
        printf("enclave_key malloc error!\n");
        return free_tee_cut_layer();
    }
    res = (cc_enclave_result_t)get_enclave_ecdhkey(context, &retval, session_id, enclave_key, enclave_key_len,
                                                   &enclave_key_len);
    if (res != CC_SUCCESS || retval != static_cast<int>(CC_SUCCESS)) {
        printf("get_enclave_ecdhkey error: %x!\n", res);
        return free_tee_cut_layer();
    }
    printf("get_enclave_ecdhkey success!\n");

    // host: get client key exchange parameter
    ckey_exchange_info = reinterpret_cast<key_exchange_t *>(malloc(sizeof(key_exchange_t)));
    if (ckey_exchange_info == NULL) {
        printf("key_exchange_info malloc error!\n");
        return free_tee_cut_layer();
    }
    memset(ckey_exchange_info, 0, sizeof(key_exchange_t));
    res = (cc_enclave_result_t)get_client_ecdhkey(ckey_exchange_info, rsa_pubkey, rsa_pubkey_len, enclave_key,
                                                  enclave_key_len, NULL, 0, &client_key, &client_key_len);
    if (res != CC_SUCCESS) {
        printf("get_client_ecdhkey error: %x!\n", res);
        return free_tee_cut_layer();
    }
    printf("get_client_ecdhkey success!\n");

    // call into enclave
    res = (cc_enclave_result_t)set_peer_ecdhkey(context, &retval, session_id, client_key, client_key_len, NULL, 0);
    if (res != CC_SUCCESS || retval != static_cast<int>(CC_SUCCESS)) {
        printf("set_peer_ecdhkey error: %x!\n", res);
        return free_tee_cut_layer();
    }
    printf("set_peer_ecdhkey success!\n");
    return res;
}

int init_tee_cut_layer(size_t batch_size, size_t featureA_dims, size_t featureB_dims, size_t output_dims,
                       float learning_rate, float loss_scale) {
    printf("init_tee_cut_layer\n");
    int  retval = 0;
    const char *path = EnclavePATH;
    char buf[BUF_LEN];

    context = reinterpret_cast<cc_enclave_t *>(calloc(1, sizeof(cc_enclave_t)));
    if (!context) {
        return CC_ERROR_OUT_OF_MEMORY;
    }
    cc_enclave_result_t res = CC_FAIL;

    printf("Create secgear enclave\n");

    char real_p[PATH_MAX];
    /* check file exists, if not exist then use absolute path */
    if (realpath(path, real_p) == NULL) {
        if (getcwd(real_p, sizeof(real_p)) == NULL) {
            printf("Cannot find enclave.sign.so");
            free(context);
            return res;
        }
        if (PATH_MAX - strlen(real_p) <= strlen("/enclave.signed.so")) {
            printf("Failed to strcat enclave.sign.so path");
            free(context);
            return res;
        }
    }
    char *final_p = reinterpret_cast<char *>(malloc(strlen(real_p) + strlen("/enclave.signed.so")));
    (void)snprintf(final_p, PATH_MAX, "%s", real_p);

    res = cc_enclave_create(final_p, SGX_ENCLAVE_TYPE, 0, SECGEAR_DEBUG_FLAG, NULL, 0, context);
    if (res != CC_SUCCESS) {
        printf("Create enclave error\n");
        free(context);
        return res;
    }
    printf("cc_enclave_create success\n");

    res = (cc_enclave_result_t)establish_secure_channel();
    if (res != CC_SUCCESS) {
        printf("Establish secure channel error!\n");
        return res;
    }

    size_t input_sizes[] = {featureA_dims, featureB_dims};
    res = init_cut_layer(context, &retval, input_sizes, 2*sizeof(size_t), output_dims, learning_rate, loss_scale, buf);
    if (res != CC_SUCCESS || retval != static_cast<int>(CC_SUCCESS)) {
        printf("Ecall enclave error\n");
    } else {
        printf("%s\n", buf);
    }
    return res;
}

std::pair<std::vector<uint8_t>, int> encrypt_client_data(std::vector<float> *plain, size_t plain_len) {
    if (!context) {
        printf("context is null\n");
        return std::pair<std::vector<uint8_t>, int>();
    }

    int ret_len = 0;
    uint8_t *out_buf = NULL;
    size_t out_buf_len = DATA_SIZE_LEN + plain_len*sizeof(float) + DATA_SIZE_LEN + GCM_TAG_LEN;
    if (ckey_exchange_info == NULL) {
        printf("get ckey_exchange_info error!\n");
        return std::pair<std::vector<uint8_t>, int>();
    }

    out_buf = reinterpret_cast<uint8_t *>(malloc(out_buf_len));
    if (out_buf == NULL) {
        printf("malloc error!\n");
        return std::pair<std::vector<uint8_t>, int>();
    }

    ret_len = secure_channel_write(ckey_exchange_info, false, reinterpret_cast<uint8_t *>(plain->data()),
                                   plain_len*sizeof(float), out_buf, out_buf_len);
    if (ret_len < 0) {
        printf("aes gcm error\n");
        return std::pair<std::vector<uint8_t>, int>();
    }

    std::vector<uint8_t> tmp(out_buf, out_buf + ret_len);
    free(out_buf);
    return std::pair<std::vector<uint8_t>, int>(tmp, ret_len);
}


std::vector<float> secure_forward_tee_cut_layer(size_t batch_size, size_t featureA_dims, size_t featureB_dims,
                                         std::vector<uint8_t> *encrypted_embA, size_t encrypted_embA_len,
                                         std::vector<uint8_t> *encrypted_embB, size_t encrypted_embB_len,
                                         size_t output_dims) {
    if (!context) {
        printf("context is null\n");
        return std::vector<float>();
    }
    cc_enclave_result_t res = CC_FAIL;
    int retval = 0;
    float *output = reinterpret_cast<float *>(malloc(sizeof(float)*batch_size*output_dims));
    if (!output) {
        free(context);
        printf("CC_ERROR_OUT_OF_MEMORY\n");
        return std::vector<float>();
    }
    for (size_t i = 0; i < batch_size*output_dims; i++) {
        output[i] = 0.0f;
    }

    res = secure_forward_cut_layer(context, &retval, session_id, batch_size, featureA_dims, featureB_dims,
                                   encrypted_embA->data(), encrypted_embA_len, sizeof(float)*featureA_dims*batch_size,
                                   encrypted_embB->data(), encrypted_embB_len, sizeof(float)*featureB_dims*batch_size,
                                   output, sizeof(float)*batch_size*output_dims);
    if (res != CC_SUCCESS || retval != static_cast<int>(CC_SUCCESS)) {
        free(context);
        printf("Ecall enclave error\n");
        return std::vector<float>();
    }

    std::vector<float> tmp(output, output + batch_size*output_dims);
    free(output);
    return tmp;
}

std::vector<float> forward_tee_cut_layer(size_t batch_size, size_t featureA_dims, size_t featureB_dims,
                                         std::vector<float> *embA, std::vector<float> *embB,
                                         size_t output_dims) {
    if (!context) {
        printf("context is null\n");
        return std::vector<float>();
    }
    cc_enclave_result_t res = CC_FAIL;
    int retval = 0;
    float *output = reinterpret_cast<float *>(malloc(sizeof(float)*batch_size*output_dims));
    if (!output) {
        free(context);
        printf("CC_ERROR_OUT_OF_MEMORY\n");
        return std::vector<float>();
    }
    for (size_t i = 0; i < batch_size*output_dims; i++) {
        output[i] = 0.0f;
    }

    res = forward_cut_layer(context, &retval, batch_size, featureA_dims, featureB_dims, embA->data(),
                            sizeof(float)*featureA_dims*batch_size, embB->data(),
                            sizeof(float)*featureB_dims*batch_size, output, sizeof(float)*batch_size*output_dims);
    if (res != CC_SUCCESS || retval != static_cast<int>(CC_SUCCESS)) {
        free(context);
        printf("Ecall enclave error\n");
        return std::vector<float>();
    }

    std::vector<float> tmp(output, output + batch_size*output_dims);
    free(output);
    return tmp;
}

std::vector<std::vector<float>> backward_tee_cut_layer(size_t batch_size, size_t featureA_dims, size_t featureB_dims,
                                                       size_t output_dims, std::vector<float> *d_output) {
    if (!context) {
        printf("context is null\n");
        return std::vector<std::vector<float>>();
    }
    cc_enclave_result_t res = CC_FAIL;
    int retval = 0;

    float *botA_delta = reinterpret_cast<float *>(calloc(batch_size*featureA_dims, sizeof(float)));
    float *botB_delta = reinterpret_cast<float *>(calloc(batch_size*featureB_dims, sizeof(float)));
    if (botA_delta == NULL || botB_delta == NULL) {
        free(context);
        printf("CC_ERROR_OUT_OF_MEMORY\n");
        return std::vector<std::vector<float>>();
    }

    res = backward_cut_layer(context, &retval, batch_size, output_dims, d_output->data(),
                             sizeof(float)*batch_size*output_dims, botA_delta,
                             sizeof(float)*batch_size*featureA_dims, botB_delta,
                             sizeof(float)*batch_size*featureB_dims);
    if (res != CC_SUCCESS || retval != static_cast<int>(CC_SUCCESS)) {
        printf("Ecall enclave error\n");
        free(context);
        return std::vector<std::vector<float>>();
    }

    std::vector<float> d_inputA(botA_delta, botA_delta + batch_size*featureA_dims);
    free(botA_delta);
    std::vector<float> d_inputB(botB_delta, botB_delta + batch_size*featureB_dims);
    free(botB_delta);
    std::vector<std::vector<float>> tmp = {d_inputA, d_inputB};
    return tmp;
}

int free_tee_cut_layer() {
    if (context != NULL) {
        cc_enclave_result_t res = CC_FAIL;
        int retval = 0;
        char buf[BUF_LEN];
        res = free_cut_layer(context, &retval, buf);
        if (res != CC_SUCCESS || retval != static_cast<int>(CC_SUCCESS)) {
            printf("Ecall enclave error\n");
        } else {
            printf("%s\n", buf);
        }
        printf("free_cut_layer success!\n");

        res = delete_enclave_ecdhkey(context, session_id);
        if (res != CC_SUCCESS) {
            printf("delete_enclave_ecdhkey error: %x!\n", res);
        }
        printf("delete_enclave_ecdhkey success!\n");

        res = delete_enclave_rsakey(context, key_id);
        if (res != CC_SUCCESS) {
            printf("delete_enclave_rsakey error: %x!\n", res);
        }
        printf("delete_enclave_rsakey success!\n");

        delete_enclave_rsakey_all(context);
        printf("delete_enclave_rsakey_all success!\n");

        res = cc_enclave_destroy(context);
        if (res != CC_SUCCESS) {
            printf("Destroy enclave error\n");
        }
        printf("cc_enclave_destroy success!\n");

        free(context);
    } else {
        printf("context is null\n");
        return -1;
    }

    if (rsa_pubkey != NULL) {
        free(rsa_pubkey);
        rsa_pubkey = NULL;
    }
    if (enclave_key != NULL) {
        free(enclave_key);
        enclave_key = NULL;
    }
    if (client_key != NULL) {
        free(client_key);
        client_key = NULL;
    }
    if (ckey_exchange_info != NULL) {
        EC_KEY_free(ckey_exchange_info->ecdh_key);
        memset(ckey_exchange_info, 0, sizeof(key_exchange_t));
        free(ckey_exchange_info);
        ckey_exchange_info = NULL;
    }

    return 0;
}
#else
int init_tee_cut_layer(size_t batch_size, size_t featureA_dims, size_t featureB_dims, size_t output_dims,
                       float learning_rate, float loss_scale) {
    printf("init_tee_cut_layer is not supported on this platform!\n");
    return -1;
}

std::vector<float> forward_tee_cut_layer(size_t batch_size, size_t featureA_dims, size_t featureB_dims,
                                         std::vector<float> *embA, std::vector<float> *embB,
                                         size_t output_dims) {
    printf("forward_tee_cut_layer is not supported on this platform!\n");
    return std::vector<float>();
}

std::vector<std::vector<float>> backward_tee_cut_layer(size_t batch_size, size_t featureA_dims, size_t featureB_dims,
                                                       size_t output_dims, std::vector<float> *d_output) {
    printf("backward_tee_cut_layer is not supported on this platform!\n");
    return std::vector<std::vector<float>>();
}

std::pair<std::vector<uint8_t>, int> encrypt_client_data(std::vector<float> *plain, size_t plain_len) {
    printf("encrypt_client_data is not supported on this platform!\n");
    return std::pair<std::vector<uint8_t>, int>();
}

std::vector<float> secure_forward_tee_cut_layer(size_t batch_size, size_t featureA_dims, size_t featureB_dims,
                                         std::vector<uint8_t> *encrypted_embA, size_t encrypted_embA_len,
                                         std::vector<uint8_t> *encrypted_embB, size_t encrypted_embB_len,
                                         size_t output_dims) {
    printf("secure_forward_tee_cut_layer is not supported on this platform!\n");
    return std::vector<float>();
}

int free_tee_cut_layer() {
    printf("free_tee_cut_layer is not supported on this platform!\n");
    return -1;
}
#endif
}  // namespace TEE
}  // namespace fl
}  // namespace mindspore
