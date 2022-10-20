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

#include <errno.h>
#include "armour/secure_protocol/enclave_call.h"

typedef struct ms_cc_enclave_PrintInfo_t {
    const char *ms_str;
}ms_cc_enclave_PrintInfo_t;

typedef struct ms_init_cut_layer_t {
    int ms_retval;
    size_t* ms_in_input_sizes;
    size_t ms_input_size;
    size_t ms_output_size;
    float ms_lr;
    float ms_loss_scale;
    char* ms_buf;
} ms_init_cut_layer_t;

typedef struct ms_free_cut_layer_t {
    int ms_retval;
    char* ms_buf;
} ms_free_cut_layer_t;

typedef struct ms_forward_cut_layer_t {
    int ms_retval;
    size_t ms_batch_size;
    size_t ms_featureA_dim;
    size_t ms_featureB_dim;
    float* ms_embA_buff;
    size_t ms_data_size_A;
    float* ms_embB_buff;
    size_t ms_data_size_B;
    float* ms_out_buff;
    size_t ms_output_size;
} ms_forward_cut_layer_t;

typedef struct ms_backward_cut_layer_t {
    int ms_retval;
    size_t ms_d_output_rows;
    size_t ms_d_output_cols;
    float* ms_d_output_buff;
    size_t ms_d_output_size;
    float* ms_d_inputA_buff;
    size_t ms_d_inputA_size;
    float* ms_d_inputB_buff;
    size_t ms_d_inputB_size;
} ms_backward_cut_layer_t;

typedef struct ms_secure_forward_cut_layer_t {
    int ms_retval;
    size_t ms_session_id;
    size_t ms_batch_size;
    size_t ms_featureA_dim;
    size_t ms_featureB_dim;
    uint8_t* ms_encrypted_embA;
    size_t ms_encrypted_embA_len;
    size_t ms_data_size_A;
    uint8_t* ms_encrypted_embB;
    size_t ms_encrypted_embB_len;
    size_t ms_data_size_B;
    float* ms_out_buff;
    size_t ms_output_size;
} ms_secure_forward_cut_layer_t;

typedef struct ms_generate_enclave_rsakey_t {
    int ms_retval;
    size_t ms_key_id;
    int ms_modulus_bits;
    size_t ms_pub_exponent;
    uint8_t* ms_rsa_pubkey;
    size_t ms_pk_in_len;
    size_t* ms_rsa_pubkey_len;
    uint8_t* ms_signature;
    size_t ms_sig_in_len;
    size_t* ms_signature_len;
} ms_generate_enclave_rsakey_t;

typedef struct ms_get_enclave_rsa_pubkey_t {
    int ms_retval;
    size_t ms_key_id;
    uint8_t* ms_rsa_pubkey;
    size_t ms_pk_in_len;
    size_t* ms_rsa_pubkey_len;
    uint8_t* ms_signature;
    size_t ms_sig_in_len;
    size_t* ms_signature_len;
} ms_get_enclave_rsa_pubkey_t;

typedef struct ms_get_enclave_rsa_encprikey_t {
    int ms_retval;
    size_t ms_key_id;
    uint8_t* ms_rsa_enc_prikey;
    size_t ms_pri_in_len;
    size_t* ms_rsa_enc_prikey_len;
} ms_get_enclave_rsa_encprikey_t;

typedef struct ms_set_enclave_rsa_prikey_t {
    int ms_retval;
    size_t ms_key_id;
    uint8_t* ms_rsa_prikey;
    size_t ms_pri_in_len;
} ms_set_enclave_rsa_prikey_t;

typedef struct ms_delete_enclave_rsakey_t {
    size_t ms_key_id;
} ms_delete_enclave_rsakey_t;

typedef struct ms_generate_enclave_ecdhkey_t {
    int ms_retval;
    size_t ms_session_id;
    size_t ms_signed_key_id;
    int ms_curve_id;
    uint8_t* ms_enclave_key;
    size_t ms_key_in_len;
    size_t* ms_enclave_key_len;
} ms_generate_enclave_ecdhkey_t;

typedef struct ms_get_enclave_ecdhkey_t {
    int ms_retval;
    size_t ms_session_id;
    uint8_t* ms_enclave_key;
    size_t ms_key_in_len;
    size_t* ms_enclave_key_len;
} ms_get_enclave_ecdhkey_t;

typedef struct ms_set_peer_ecdhkey_t {
    int ms_retval;
    size_t ms_session_id;
    uint8_t* ms_peer_key;
    size_t ms_key_in_len;
    uint8_t* ms_verify_key;
    size_t ms_verify_key_len;
} ms_set_peer_ecdhkey_t;

typedef struct ms_delete_enclave_ecdhkey_t {
    size_t ms_session_id;
} ms_delete_enclave_ecdhkey_t;

typedef struct ms_ocall_EIGEN_CPUID_t {
    int32_t* ms_abcd;
    uint32_t ms_func;
    uint32_t ms_id;
} ms_ocall_EIGEN_CPUID_t;

typedef struct ms_pthread_wait_timeout_ocall_t {
    int ms_retval;
    uint64_t ms_waiter;
    uint64_t ms_timeout;
} ms_pthread_wait_timeout_ocall_t;

typedef struct ms_pthread_create_ocall_t {
    int ms_retval;
    uint64_t ms_self;
} ms_pthread_create_ocall_t;

typedef struct ms_pthread_wakeup_ocall_t {
    int ms_retval;
    uint64_t ms_waiter;
} ms_pthread_wakeup_ocall_t;

typedef struct ms_sgx_oc_cpuidex_t {
    int* ms_cpuinfo;
    int ms_leaf;
    int ms_subleaf;
} ms_sgx_oc_cpuidex_t;

typedef struct ms_sgx_thread_wait_untrusted_event_ocall_t {
    int ms_retval;
    const void* ms_self;
} ms_sgx_thread_wait_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_set_untrusted_event_ocall_t {
    int ms_retval;
    const void* ms_waiter;
} ms_sgx_thread_set_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_setwait_untrusted_events_ocall_t {
    int ms_retval;
    const void* ms_waiter;
    const void* ms_self;
} ms_sgx_thread_setwait_untrusted_events_ocall_t;

typedef struct ms_sgx_thread_set_multiple_untrusted_events_ocall_t {
    int ms_retval;
    const void** ms_waiters;
    size_t ms_total;
} ms_sgx_thread_set_multiple_untrusted_events_ocall_t;

typedef struct ms_ocall_print_string_t {
    const char* ms_str;
} ms_ocall_print_string_t;

typedef struct ms_u_sgxssl_ftime_t {
    void* ms_timeptr;
    uint32_t ms_timeb_len;
} ms_u_sgxssl_ftime_t;

typedef struct ms_ocall_cc_read_t {
    int ms_retval;
    int ms_fd;
    void* ms_buf;
    size_t ms_buf_len;
} ms_ocall_cc_read_t;

typedef struct ms_ocall_cc_write_t {
    int ms_retval;
    int ms_fd;
    const void* ms_buf;
    size_t ms_buf_len;
} ms_ocall_cc_write_t;

typedef struct ms_ocall_cc_getenv_t {
    int ms_retval;
    const char* ms_name;
    size_t ms_name_len;
    void* ms_buf;
    int ms_buf_len;
    int* ms_need_len;
} ms_ocall_cc_getenv_t;

typedef struct ms_ocall_cc_fopen_t {
    uint64_t ms_retval;
    const char* ms_filename;
    size_t ms_filename_len;
    const char* ms_mode;
    size_t ms_mode_len;
} ms_ocall_cc_fopen_t;

typedef struct ms_ocall_cc_fclose_t {
    int ms_retval;
    uint64_t ms_fp;
} ms_ocall_cc_fclose_t;

typedef struct ms_ocall_cc_ferror_t {
    int ms_retval;
    uint64_t ms_fp;
} ms_ocall_cc_ferror_t;

typedef struct ms_ocall_cc_feof_t {
    int ms_retval;
    uint64_t ms_fp;
} ms_ocall_cc_feof_t;

typedef struct ms_ocall_cc_fflush_t {
    int ms_retval;
    uint64_t ms_fp;
} ms_ocall_cc_fflush_t;

typedef struct ms_ocall_cc_ftell_t {
    int64_t ms_retval;
    uint64_t ms_fp;
} ms_ocall_cc_ftell_t;

typedef struct ms_ocall_cc_fseek_t {
    int ms_retval;
    uint64_t ms_fp;
    int64_t ms_offset;
    int ms_origin;
} ms_ocall_cc_fseek_t;

typedef struct ms_ocall_cc_fread_t {
    size_t ms_retval;
    void* ms_buf;
    size_t ms_total_size;
    size_t ms_element_size;
    size_t ms_cnt;
    uint64_t ms_fp;
} ms_ocall_cc_fread_t;

typedef struct ms_ocall_cc_fwrite_t {
    size_t ms_retval;
    const void* ms_buf;
    size_t ms_total_size;
    size_t ms_element_size;
    size_t ms_cnt;
    uint64_t ms_fp;
} ms_ocall_cc_fwrite_t;

typedef struct ms_ocall_cc_fgets_t {
    int ms_retval;
    char* ms_str;
    int ms_max_cnt;
    uint64_t ms_fp;
} ms_ocall_cc_fgets_t;

typedef struct ms_ocall_cc_fputs_t {
    int ms_retval;
    const char* ms_str;
    size_t ms_total_size;
    uint64_t ms_fp;
} ms_ocall_cc_fputs_t;

static cc_enclave_result_t cutlayer_ocall_EIGEN_CPUID(void* pms) {
    ms_ocall_EIGEN_CPUID_t* ms = SGX_CAST(ms_ocall_EIGEN_CPUID_t*, pms);
    ocall_EIGEN_CPUID(ms->ms_abcd, ms->ms_func, ms->ms_id);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_pthread_wait_timeout_ocall(void* pms) {
    ms_pthread_wait_timeout_ocall_t* ms = SGX_CAST(ms_pthread_wait_timeout_ocall_t*, pms);
    ms->ms_retval = pthread_wait_timeout_ocall(ms->ms_waiter, ms->ms_timeout);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_pthread_create_ocall(void* pms) {
    ms_pthread_create_ocall_t* ms = SGX_CAST(ms_pthread_create_ocall_t*, pms);
    ms->ms_retval = pthread_create_ocall(ms->ms_self);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_pthread_wakeup_ocall(void* pms) {
    ms_pthread_wakeup_ocall_t* ms = SGX_CAST(ms_pthread_wakeup_ocall_t*, pms);
    ms->ms_retval = pthread_wakeup_ocall(ms->ms_waiter);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_sgx_oc_cpuidex(void* pms) {
    ms_sgx_oc_cpuidex_t* ms = SGX_CAST(ms_sgx_oc_cpuidex_t*, pms);
    sgx_oc_cpuidex(ms->ms_cpuinfo, ms->ms_leaf, ms->ms_subleaf);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_sgx_thread_wait_untrusted_event_ocall(void* pms) {
    ms_sgx_thread_wait_untrusted_event_ocall_t* ms = SGX_CAST(ms_sgx_thread_wait_untrusted_event_ocall_t*, pms);
    ms->ms_retval = sgx_thread_wait_untrusted_event_ocall(ms->ms_self);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_sgx_thread_set_untrusted_event_ocall(void* pms) {
    ms_sgx_thread_set_untrusted_event_ocall_t* ms = SGX_CAST(ms_sgx_thread_set_untrusted_event_ocall_t*, pms);
    ms->ms_retval = sgx_thread_set_untrusted_event_ocall(ms->ms_waiter);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_sgx_thread_setwait_untrusted_events_ocall(void* pms) {
    ms_sgx_thread_setwait_untrusted_events_ocall_t* ms = SGX_CAST(ms_sgx_thread_setwait_untrusted_events_ocall_t*, pms);
    ms->ms_retval = sgx_thread_setwait_untrusted_events_ocall(ms->ms_waiter, ms->ms_self);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_sgx_thread_set_multiple_untrusted_events_ocall(void* pms) {
    ms_sgx_thread_set_multiple_untrusted_events_ocall_t* ms =
    SGX_CAST(ms_sgx_thread_set_multiple_untrusted_events_ocall_t*, pms);
    ms->ms_retval = sgx_thread_set_multiple_untrusted_events_ocall(ms->ms_waiters, ms->ms_total);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_print_string(void* pms) {
    ms_ocall_print_string_t* ms = SGX_CAST(ms_ocall_print_string_t*, pms);
    ocall_print_string(ms->ms_str);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_u_sgxssl_ftime(void* pms) {
    ms_u_sgxssl_ftime_t* ms = SGX_CAST(ms_u_sgxssl_ftime_t*, pms);
    u_sgxssl_ftime(ms->ms_timeptr, ms->ms_timeb_len);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_read(void* pms) {
    ms_ocall_cc_read_t* ms = SGX_CAST(ms_ocall_cc_read_t*, pms);
    ms->ms_retval = ocall_cc_read(ms->ms_fd, ms->ms_buf, ms->ms_buf_len);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_write(void* pms) {
    ms_ocall_cc_write_t* ms = SGX_CAST(ms_ocall_cc_write_t*, pms);
    ms->ms_retval = ocall_cc_write(ms->ms_fd, ms->ms_buf, ms->ms_buf_len);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_getenv(void* pms) {
    ms_ocall_cc_getenv_t* ms = SGX_CAST(ms_ocall_cc_getenv_t*, pms);
    ms->ms_retval = ocall_cc_getenv(ms->ms_name, ms->ms_name_len, ms->ms_buf, ms->ms_buf_len, ms->ms_need_len);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_fopen(void* pms) {
    ms_ocall_cc_fopen_t* ms = SGX_CAST(ms_ocall_cc_fopen_t*, pms);
    ms->ms_retval = ocall_cc_fopen(ms->ms_filename, ms->ms_filename_len, ms->ms_mode, ms->ms_mode_len);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_fclose(void* pms) {
    ms_ocall_cc_fclose_t* ms = SGX_CAST(ms_ocall_cc_fclose_t*, pms);
    ms->ms_retval = ocall_cc_fclose(ms->ms_fp);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_ferror(void* pms) {
    ms_ocall_cc_ferror_t* ms = SGX_CAST(ms_ocall_cc_ferror_t*, pms);
    ms->ms_retval = ocall_cc_ferror(ms->ms_fp);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_feof(void* pms) {
    ms_ocall_cc_feof_t* ms = SGX_CAST(ms_ocall_cc_feof_t*, pms);
    ms->ms_retval = ocall_cc_feof(ms->ms_fp);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_fflush(void* pms) {
    ms_ocall_cc_fflush_t* ms = SGX_CAST(ms_ocall_cc_fflush_t*, pms);
    ms->ms_retval = ocall_cc_fflush(ms->ms_fp);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_ftell(void* pms) {
    ms_ocall_cc_ftell_t* ms = SGX_CAST(ms_ocall_cc_ftell_t*, pms);
    ms->ms_retval = ocall_cc_ftell(ms->ms_fp);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_fseek(void* pms) {
    ms_ocall_cc_fseek_t* ms = SGX_CAST(ms_ocall_cc_fseek_t*, pms);
    ms->ms_retval = ocall_cc_fseek(ms->ms_fp, ms->ms_offset, ms->ms_origin);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_fread(void* pms) {
    ms_ocall_cc_fread_t* ms = SGX_CAST(ms_ocall_cc_fread_t*, pms);
    ms->ms_retval = ocall_cc_fread(ms->ms_buf, ms->ms_total_size, ms->ms_element_size, ms->ms_cnt, ms->ms_fp);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_fwrite(void* pms) {
    ms_ocall_cc_fwrite_t* ms = SGX_CAST(ms_ocall_cc_fwrite_t*, pms);
    ms->ms_retval = ocall_cc_fwrite(ms->ms_buf, ms->ms_total_size, ms->ms_element_size, ms->ms_cnt, ms->ms_fp);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_fgets(void* pms) {
    ms_ocall_cc_fgets_t* ms = SGX_CAST(ms_ocall_cc_fgets_t*, pms);
    ms->ms_retval = ocall_cc_fgets(ms->ms_str, ms->ms_max_cnt, ms->ms_fp);

    return CC_SUCCESS;
}

static cc_enclave_result_t cutlayer_ocall_cc_fputs(void* pms) {
    ms_ocall_cc_fputs_t* ms = SGX_CAST(ms_ocall_cc_fputs_t*, pms);
    ms->ms_retval = ocall_cc_fputs(ms->ms_str, ms->ms_total_size, ms->ms_fp);

    return CC_SUCCESS;
}

static cc_enclave_result_t ocall_cc_enclave_PrintInfo(void *pms) {
    ms_cc_enclave_PrintInfo_t *ms = SGX_CAST(ms_cc_enclave_PrintInfo_t*, pms);
    cc_enclave_PrintInfo(ms->ms_str);
    return CC_SUCCESS;
}

static struct {
    size_t nr_ocall;
    void * table[26];
} ocall_table_cutlayer = {
    .nr_ocall = 26,
    .table = {
        reinterpret_cast<void*>(ocall_cc_enclave_PrintInfo),
        reinterpret_cast<void*>(cutlayer_ocall_EIGEN_CPUID),
        reinterpret_cast<void*>(cutlayer_pthread_wait_timeout_ocall),
        reinterpret_cast<void*>(cutlayer_pthread_create_ocall),
        reinterpret_cast<void*>(cutlayer_pthread_wakeup_ocall),
        reinterpret_cast<void*>(cutlayer_sgx_oc_cpuidex),
        reinterpret_cast<void*>(cutlayer_sgx_thread_wait_untrusted_event_ocall),
        reinterpret_cast<void*>(cutlayer_sgx_thread_set_untrusted_event_ocall),
        reinterpret_cast<void*>(cutlayer_sgx_thread_setwait_untrusted_events_ocall),
        reinterpret_cast<void*>(cutlayer_sgx_thread_set_multiple_untrusted_events_ocall),
        reinterpret_cast<void*>(cutlayer_ocall_print_string),
        reinterpret_cast<void*>(cutlayer_u_sgxssl_ftime),
        reinterpret_cast<void*>(cutlayer_ocall_cc_read),
        reinterpret_cast<void*>(cutlayer_ocall_cc_write),
        reinterpret_cast<void*>(cutlayer_ocall_cc_getenv),
        reinterpret_cast<void*>(cutlayer_ocall_cc_fopen),
        reinterpret_cast<void*>(cutlayer_ocall_cc_fclose),
        reinterpret_cast<void*>(cutlayer_ocall_cc_ferror),
        reinterpret_cast<void*>(cutlayer_ocall_cc_feof),
        reinterpret_cast<void*>(cutlayer_ocall_cc_fflush),
        reinterpret_cast<void*>(cutlayer_ocall_cc_ftell),
        reinterpret_cast<void*>(cutlayer_ocall_cc_fseek),
        reinterpret_cast<void*>(cutlayer_ocall_cc_fread),
        reinterpret_cast<void*>(cutlayer_ocall_cc_fwrite),
        reinterpret_cast<void*>(cutlayer_ocall_cc_fgets),
        reinterpret_cast<void*>(cutlayer_ocall_cc_fputs),
    }
};
cc_enclave_result_t init_cut_layer(cc_enclave_t *enclave, int* retval, size_t* in_input_sizes, size_t input_size,
                                   size_t output_size, float lr, float loss_scale, char* buf) {
    cc_enclave_result_t result;
    ms_init_cut_layer_t ms;
    ms.ms_in_input_sizes = in_input_sizes;
    ms.ms_input_size = input_size;
    ms.ms_output_size = output_size;
    ms.ms_lr = lr;
    ms.ms_loss_scale = loss_scale;
    ms.ms_buf = buf;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc || !enclave->list_ops_node->ops_desc->ops ||
        !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(enclave, 0, NULL, 0, NULL,
                                                                     0, &ms, &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    if (result == CC_SUCCESS && retval) {
        *retval = ms.ms_retval;
    }
    return result;
}

cc_enclave_result_t free_cut_layer(cc_enclave_t *enclave, int* retval, char* buf) {
    cc_enclave_result_t result;
    ms_free_cut_layer_t ms;
    ms.ms_buf = buf;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc || !enclave->list_ops_node->ops_desc->ops ||
        !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(enclave, 1, NULL, 0, NULL,
                                                                     0, &ms, &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    if (result == CC_SUCCESS && retval) {
        *retval = ms.ms_retval;
    }
    return result;
}

cc_enclave_result_t forward_cut_layer(cc_enclave_t *enclave, int* retval, size_t batch_size, size_t featureA_dim,
                                      size_t featureB_dim, float* embA_buff, size_t data_size_A, float* embB_buff,
                                      size_t data_size_B, float* out_buff, size_t output_size) {
    cc_enclave_result_t result;
    ms_forward_cut_layer_t ms;
    ms.ms_batch_size = batch_size;
    ms.ms_featureA_dim = featureA_dim;
    ms.ms_featureB_dim = featureB_dim;
    ms.ms_embA_buff = embA_buff;
    ms.ms_data_size_A = data_size_A;
    ms.ms_embB_buff = embB_buff;
    ms.ms_data_size_B = data_size_B;
    ms.ms_out_buff = out_buff;
    ms.ms_output_size = output_size;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc ||
    !enclave->list_ops_node->ops_desc->ops ||
    !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(
    enclave,
    2,
    NULL,
    0,
    NULL,
    0,
    &ms,
    &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    if (result == CC_SUCCESS && retval) {
        *retval = ms.ms_retval;
    }
    return result;
}

cc_enclave_result_t backward_cut_layer(cc_enclave_t *enclave, int* retval, size_t d_output_rows, size_t d_output_cols,
                                       float* d_output_buff, size_t d_output_size, float* d_inputA_buff,
                                       size_t d_inputA_size, float* d_inputB_buff, size_t d_inputB_size) {
    cc_enclave_result_t result;
    ms_backward_cut_layer_t ms;
    ms.ms_d_output_rows = d_output_rows;
    ms.ms_d_output_cols = d_output_cols;
    ms.ms_d_output_buff = d_output_buff;
    ms.ms_d_output_size = d_output_size;
    ms.ms_d_inputA_buff = d_inputA_buff;
    ms.ms_d_inputA_size = d_inputA_size;
    ms.ms_d_inputB_buff = d_inputB_buff;
    ms.ms_d_inputB_size = d_inputB_size;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc ||
    !enclave->list_ops_node->ops_desc->ops ||
    !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(
    enclave,
    3,
    NULL,
    0,
    NULL,
    0,
    &ms,
    &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    if (result == CC_SUCCESS && retval) {
        *retval = ms.ms_retval;
    }
    return result;
}

cc_enclave_result_t secure_forward_cut_layer(cc_enclave_t *enclave, int* retval, size_t session_id, size_t batch_size,
                                             size_t featureA_dim, size_t featureB_dim, uint8_t* encrypted_embA,
                                             size_t encrypted_embA_len, size_t data_size_A, uint8_t* encrypted_embB,
                                             size_t encrypted_embB_len, size_t data_size_B, float* out_buff,
                                             size_t output_size) {
    cc_enclave_result_t result;
    ms_secure_forward_cut_layer_t ms;
    ms.ms_session_id = session_id;
    ms.ms_batch_size = batch_size;
    ms.ms_featureA_dim = featureA_dim;
    ms.ms_featureB_dim = featureB_dim;
    ms.ms_encrypted_embA = encrypted_embA;
    ms.ms_encrypted_embA_len = encrypted_embA_len;
    ms.ms_data_size_A = data_size_A;
    ms.ms_encrypted_embB = encrypted_embB;
    ms.ms_encrypted_embB_len = encrypted_embB_len;
    ms.ms_data_size_B = data_size_B;
    ms.ms_out_buff = out_buff;
    ms.ms_output_size = output_size;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc ||
    !enclave->list_ops_node->ops_desc->ops ||
    !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(
    enclave,
    4,
    NULL,
    0,
    NULL,
    0,
    &ms,
    &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    if (result == CC_SUCCESS && retval) {
        *retval = ms.ms_retval;
    }
    return result;
}

cc_enclave_result_t generate_enclave_rsakey(cc_enclave_t *enclave, int* retval, size_t key_id, int modulus_bits,
                                            size_t pub_exponent, uint8_t* rsa_pubkey, size_t pk_in_len,
                                            size_t* rsa_pubkey_len, uint8_t* signature, size_t sig_in_len,
                                            size_t* signature_len) {
    cc_enclave_result_t result;
    ms_generate_enclave_rsakey_t ms;
    ms.ms_key_id = key_id;
    ms.ms_modulus_bits = modulus_bits;
    ms.ms_pub_exponent = pub_exponent;
    ms.ms_rsa_pubkey = rsa_pubkey;
    ms.ms_pk_in_len = pk_in_len;
    ms.ms_rsa_pubkey_len = rsa_pubkey_len;
    ms.ms_signature = signature;
    ms.ms_sig_in_len = sig_in_len;
    ms.ms_signature_len = signature_len;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc ||
    !enclave->list_ops_node->ops_desc->ops ||
    !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(
    enclave,
    5,
    NULL,
    0,
    NULL,
    0,
    &ms,
    &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    if (result == CC_SUCCESS && retval) {
        *retval = ms.ms_retval;
    }
    return result;
}

cc_enclave_result_t get_enclave_rsa_pubkey(cc_enclave_t *enclave, int* retval, size_t key_id, uint8_t* rsa_pubkey,
                                           size_t pk_in_len, size_t* rsa_pubkey_len, uint8_t* signature,
                                           size_t sig_in_len, size_t* signature_len) {
    cc_enclave_result_t result;
    ms_get_enclave_rsa_pubkey_t ms;
    ms.ms_key_id = key_id;
    ms.ms_rsa_pubkey = rsa_pubkey;
    ms.ms_pk_in_len = pk_in_len;
    ms.ms_rsa_pubkey_len = rsa_pubkey_len;
    ms.ms_signature = signature;
    ms.ms_sig_in_len = sig_in_len;
    ms.ms_signature_len = signature_len;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
    return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc ||
    !enclave->list_ops_node->ops_desc->ops ||
    !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(
    enclave,
    6,
    NULL,
    0,
    NULL,
    0,
    &ms,
    &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    if (result == CC_SUCCESS && retval) {
        *retval = ms.ms_retval;
    }
    return result;
}

cc_enclave_result_t get_enclave_rsa_encprikey(cc_enclave_t *enclave, int* retval, size_t key_id,
                                              uint8_t* rsa_enc_prikey, size_t pri_in_len,
                                              size_t* rsa_enc_prikey_len) {
    cc_enclave_result_t result;
    ms_get_enclave_rsa_encprikey_t ms;
    ms.ms_key_id = key_id;
    ms.ms_rsa_enc_prikey = rsa_enc_prikey;
    ms.ms_pri_in_len = pri_in_len;
    ms.ms_rsa_enc_prikey_len = rsa_enc_prikey_len;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc ||
    !enclave->list_ops_node->ops_desc->ops ||
    !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(
    enclave,
    7,
    NULL,
    0,
    NULL,
    0,
    &ms,
    &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    if (result == CC_SUCCESS && retval) {
        *retval = ms.ms_retval;
    }
    return result;
}

cc_enclave_result_t set_enclave_rsa_prikey(cc_enclave_t *enclave, int* retval, size_t key_id, uint8_t* rsa_prikey,
                                           size_t pri_in_len) {
    cc_enclave_result_t result;
    ms_set_enclave_rsa_prikey_t ms;
    ms.ms_key_id = key_id;
    ms.ms_rsa_prikey = rsa_prikey;
    ms.ms_pri_in_len = pri_in_len;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc ||
    !enclave->list_ops_node->ops_desc->ops ||
    !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(
    enclave,
    8,
    NULL,
    0,
    NULL,
    0,
    &ms,
    &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    if (result == CC_SUCCESS && retval) {
        *retval = ms.ms_retval;
    }
    return result;
}

cc_enclave_result_t delete_enclave_rsakey(cc_enclave_t *enclave, size_t key_id) {
    cc_enclave_result_t result;
    ms_delete_enclave_rsakey_t ms;
    ms.ms_key_id = key_id;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc ||
    !enclave->list_ops_node->ops_desc->ops ||
    !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(
    enclave,
    9,
    NULL,
    0,
    NULL,
    0,
    &ms,
    &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    return result;
}

cc_enclave_result_t delete_enclave_rsakey_all(cc_enclave_t *enclave) {
    cc_enclave_result_t result;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave || !enclave->list_ops_node || !enclave->list_ops_node->ops_desc ||
    !enclave->list_ops_node->ops_desc->ops ||
    !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(
    enclave,
    10,
    NULL,
    0,
    NULL,
    0,
    NULL,
    &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    return result;
}

cc_enclave_result_t generate_enclave_ecdhkey(cc_enclave_t *enclave, int* retval, size_t session_id,
                                             size_t signed_key_id, int curve_id, uint8_t* enclave_key,
                                             size_t key_in_len, size_t* enclave_key_len) {
    cc_enclave_result_t result;
    ms_generate_enclave_ecdhkey_t ms;
    ms.ms_session_id = session_id;
    ms.ms_signed_key_id = signed_key_id;
    ms.ms_curve_id = curve_id;
    ms.ms_enclave_key = enclave_key;
    ms.ms_key_in_len = key_in_len;
    ms.ms_enclave_key_len = enclave_key_len;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc ||
    !enclave->list_ops_node->ops_desc->ops ||
    !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(
    enclave,
    11,
    NULL,
    0,
    NULL,
    0,
    &ms,
    &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    if (result == CC_SUCCESS && retval) {
        *retval = ms.ms_retval;
    }
    return result;
}

cc_enclave_result_t get_enclave_ecdhkey(cc_enclave_t *enclave, int* retval, size_t session_id, uint8_t* enclave_key,
                                        size_t key_in_len, size_t* enclave_key_len) {
    cc_enclave_result_t result;
    ms_get_enclave_ecdhkey_t ms;
    ms.ms_session_id = session_id;
    ms.ms_enclave_key = enclave_key;
    ms.ms_key_in_len = key_in_len;
    ms.ms_enclave_key_len = enclave_key_len;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc ||
    !enclave->list_ops_node->ops_desc->ops ||
    !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(
    enclave,
    12,
    NULL,
    0,
    NULL,
    0,
    &ms,
    &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    if (result == CC_SUCCESS && retval) {
        *retval = ms.ms_retval;
    }
    return result;
}

cc_enclave_result_t set_peer_ecdhkey(cc_enclave_t *enclave, int* retval, size_t session_id, uint8_t* peer_key,
                                     size_t key_in_len, uint8_t* verify_key, size_t verify_key_len) {
    cc_enclave_result_t result;
    ms_set_peer_ecdhkey_t ms;
    ms.ms_session_id = session_id;
    ms.ms_peer_key = peer_key;
    ms.ms_key_in_len = key_in_len;
    ms.ms_verify_key = verify_key;
    ms.ms_verify_key_len = verify_key_len;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc ||
    !enclave->list_ops_node->ops_desc->ops ||
    !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(
    enclave,
    13,
    NULL,
    0,
    NULL,
    0,
    &ms,
    &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    if (result == CC_SUCCESS && retval) {
        *retval = ms.ms_retval;
    }
    return result;
}

cc_enclave_result_t delete_enclave_ecdhkey(cc_enclave_t *enclave, size_t session_id) {
    cc_enclave_result_t result;
    ms_delete_enclave_ecdhkey_t ms;
    ms.ms_session_id = session_id;
    if (!enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    if (pthread_rwlock_rdlock(&enclave->rwlock)) {
        return CC_ERROR_BUSY;
    }
    if (!enclave->list_ops_node || !enclave->list_ops_node->ops_desc ||
    !enclave->list_ops_node->ops_desc->ops ||
    !enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave) {
        return CC_ERROR_BAD_PARAMETERS;
    }
    result = enclave->list_ops_node->ops_desc->ops->cc_ecall_enclave(
    enclave,
    14,
    NULL,
    0,
    NULL,
    0,
    &ms,
    &ocall_table_cutlayer);
    pthread_rwlock_unlock(&enclave->rwlock);

    return result;
}
