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

#ifndef CUTLAYER_U_H__
#define CUTLAYER_U_H__

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <wchar.h>
#include <stddef.h>
#include <string.h>
#include <secGear/ocall_log.h>
#include <secGear/enclave_internal.h>
#include <secGear/secgear_urts.h>
#include "armour/secure_protocol/secure_channel.h"

#define SGX_CAST(type, item) ((type)(item))

#define SGX_UBRIDGE(attr, fname, args...) attr fname args
#define SGX_NOCONVENTION /* Empty.  No calling convention specified. */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef OCALL_EIGEN_CPUID_DEFINED__
#define OCALL_EIGEN_CPUID_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_EIGEN_CPUID, (int32_t* abcd, uint32_t func, uint32_t id));
#endif
#ifndef PTHREAD_WAIT_TIMEOUT_OCALL_DEFINED__
#define PTHREAD_WAIT_TIMEOUT_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, pthread_wait_timeout_ocall, (uint64_t waiter, uint64_t timeout));
#endif
#ifndef PTHREAD_CREATE_OCALL_DEFINED__
#define PTHREAD_CREATE_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, pthread_create_ocall, (uint64_t self));
#endif
#ifndef PTHREAD_WAKEUP_OCALL_DEFINED__
#define PTHREAD_WAKEUP_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, pthread_wakeup_ocall, (uint64_t waiter));
#endif
#ifndef SGX_OC_CPUIDEX_DEFINED__
#define SGX_OC_CPUIDEX_DEFINED__
void SGX_UBRIDGE(SGX_CDECL, sgx_oc_cpuidex, (int cpuinfo[4], int leaf, int subleaf));
#endif
#ifndef SGX_THREAD_WAIT_UNTRUSTED_EVENT_OCALL_DEFINED__
#define SGX_THREAD_WAIT_UNTRUSTED_EVENT_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, sgx_thread_wait_untrusted_event_ocall, (const void* self));
#endif
#ifndef SGX_THREAD_SET_UNTRUSTED_EVENT_OCALL_DEFINED__
#define SGX_THREAD_SET_UNTRUSTED_EVENT_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, sgx_thread_set_untrusted_event_ocall, (const void* waiter));
#endif
#ifndef SGX_THREAD_SETWAIT_UNTRUSTED_EVENTS_OCALL_DEFINED__
#define SGX_THREAD_SETWAIT_UNTRUSTED_EVENTS_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, sgx_thread_setwait_untrusted_events_ocall, (const void* waiter, const void* self));
#endif
#ifndef SGX_THREAD_SET_MULTIPLE_UNTRUSTED_EVENTS_OCALL_DEFINED__
#define SGX_THREAD_SET_MULTIPLE_UNTRUSTED_EVENTS_OCALL_DEFINED__
int SGX_UBRIDGE(SGX_CDECL, sgx_thread_set_multiple_untrusted_events_ocall, (const void** waiters, size_t total));
#endif
#ifndef OCALL_PRINT_STRING_DEFINED__
#define OCALL_PRINT_STRING_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_print_string, (const char* str));
#endif
#ifndef U_SGXSSL_FTIME_DEFINED__
#define U_SGXSSL_FTIME_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, u_sgxssl_ftime, (void* timeptr, uint32_t timeb_len));
#endif
#ifndef OCALL_CC_READ_DEFINED__
#define OCALL_CC_READ_DEFINED__
int SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_read, (int fd, void* buf, size_t buf_len));
#endif
#ifndef OCALL_CC_WRITE_DEFINED__
#define OCALL_CC_WRITE_DEFINED__
int SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_write, (int fd, const void* buf, size_t buf_len));
#endif
#ifndef OCALL_CC_GETENV_DEFINED__
#define OCALL_CC_GETENV_DEFINED__
int SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_getenv, (const char* name, size_t name_len,
                                                    void* buf, int buf_len, int* need_len));
#endif
#ifndef OCALL_CC_FOPEN_DEFINED__
#define OCALL_CC_FOPEN_DEFINED__
uint64_t SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_fopen, (const char* filename, size_t filename_len,
                                                        const char* mode, size_t mode_len));
#endif
#ifndef OCALL_CC_FCLOSE_DEFINED__
#define OCALL_CC_FCLOSE_DEFINED__
int SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_fclose, (uint64_t fp));
#endif
#ifndef OCALL_CC_FERROR_DEFINED__
#define OCALL_CC_FERROR_DEFINED__
int SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_ferror, (uint64_t fp));
#endif
#ifndef OCALL_CC_FEOF_DEFINED__
#define OCALL_CC_FEOF_DEFINED__
int SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_feof, (uint64_t fp));
#endif
#ifndef OCALL_CC_FFLUSH_DEFINED__
#define OCALL_CC_FFLUSH_DEFINED__
int SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_fflush, (uint64_t fp));
#endif
#ifndef OCALL_CC_FTELL_DEFINED__
#define OCALL_CC_FTELL_DEFINED__
int64_t SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_ftell, (uint64_t fp));
#endif
#ifndef OCALL_CC_FSEEK_DEFINED__
#define OCALL_CC_FSEEK_DEFINED__
int SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_fseek, (uint64_t fp, int64_t offset, int origin));
#endif
#ifndef OCALL_CC_FREAD_DEFINED__
#define OCALL_CC_FREAD_DEFINED__
size_t SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_fread, (void* buf, size_t total_size,
                                                      size_t element_size, size_t cnt, uint64_t fp));
#endif
#ifndef OCALL_CC_FWRITE_DEFINED__
#define OCALL_CC_FWRITE_DEFINED__
size_t SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_fwrite, (const void* buf, size_t total_size,
                                                       size_t element_size, size_t cnt, uint64_t fp));
#endif
#ifndef OCALL_CC_FGETS_DEFINED__
#define OCALL_CC_FGETS_DEFINED__
int SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_fgets, (char* str, int max_cnt, uint64_t fp));
#endif
#ifndef OCALL_CC_FPUTS_DEFINED__
#define OCALL_CC_FPUTS_DEFINED__
int SGX_UBRIDGE(SGX_NOCONVENTION, ocall_cc_fputs, (const char* str, size_t total_size, uint64_t fp));
#endif
#ifndef SECGEAR_PRINTINFO_DEFINED__
#define SECGEAR_PRINTINFO_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, cc_enclave_PrintInfo, (const char *str));
#endif


cc_enclave_result_t init_cut_layer(cc_enclave_t *enclave, int* retval, size_t* in_input_sizes, size_t input_size,
                                   size_t output_size, float lr, float loss_scale, char* buf);
cc_enclave_result_t free_cut_layer(cc_enclave_t *enclave, int* retval, char* buf);
cc_enclave_result_t forward_cut_layer(cc_enclave_t *enclave, int* retval, size_t batch_size, size_t featureA_dim,
                                      size_t featureB_dim, float* embA_buff, size_t data_size_A, float* embB_buff,
                                      size_t data_size_B, float* out_buff, size_t output_size);
cc_enclave_result_t backward_cut_layer(cc_enclave_t *enclave, int* retval, size_t d_output_rows, size_t d_output_cols,
                                       float* d_output_buff, size_t d_output_size, float* d_inputA_buff,
                                       size_t d_inputA_size, float* d_inputB_buff, size_t d_inputB_size);
cc_enclave_result_t secure_forward_cut_layer(cc_enclave_t *enclave, int* retval, size_t session_id, size_t batch_size,
                                             size_t featureA_dim, size_t featureB_dim, uint8_t* encrypted_embA,
                                             size_t encrypted_embA_len, size_t data_size_A, uint8_t* encrypted_embB,
                                             size_t encrypted_embB_len, size_t data_size_B,
                                             float* out_buff, size_t output_size);
cc_enclave_result_t generate_enclave_rsakey(cc_enclave_t *enclave, int* retval, size_t key_id, int modulus_bits,
                                            size_t pub_exponent, uint8_t* rsa_pubkey, size_t pk_in_len,
                                            size_t* rsa_pubkey_len, uint8_t* signature,
                                            size_t sig_in_len, size_t* signature_len);
cc_enclave_result_t get_enclave_rsa_pubkey(cc_enclave_t *enclave, int* retval, size_t key_id, uint8_t* rsa_pubkey,
                                           size_t pk_in_len, size_t* rsa_pubkey_len, uint8_t* signature,
                                           size_t sig_in_len, size_t* signature_len);
cc_enclave_result_t get_enclave_rsa_encprikey(cc_enclave_t *enclave, int* retval, size_t key_id,
                                              uint8_t* rsa_enc_prikey, size_t pri_in_len, size_t* rsa_enc_prikey_len);
cc_enclave_result_t set_enclave_rsa_prikey(cc_enclave_t *enclave, int* retval, size_t key_id,
                                           uint8_t* rsa_prikey, size_t pri_in_len);
cc_enclave_result_t delete_enclave_rsakey(cc_enclave_t *enclave, size_t key_id);
cc_enclave_result_t delete_enclave_rsakey_all(cc_enclave_t *enclave);
cc_enclave_result_t generate_enclave_ecdhkey(cc_enclave_t *enclave, int* retval, size_t session_id,
                                             size_t signed_key_id, int curve_id, uint8_t* enclave_key,
                                             size_t key_in_len, size_t* enclave_key_len);
cc_enclave_result_t get_enclave_ecdhkey(cc_enclave_t *enclave, int* retval, size_t session_id, uint8_t* enclave_key,
                                        size_t key_in_len, size_t* enclave_key_len);
cc_enclave_result_t set_peer_ecdhkey(cc_enclave_t *enclave, int* retval, size_t session_id, uint8_t* peer_key,
                                     size_t key_in_len, uint8_t* verify_key, size_t verify_key_len);
cc_enclave_result_t delete_enclave_ecdhkey(cc_enclave_t *enclave, size_t session_id);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
