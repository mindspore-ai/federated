/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "common/utils/python_adapter.h"
#include "python/fl_context.h"
#include "python/federated_job.h"
#include "worker/kernel/start_fl_job_kernel.h"

namespace py = pybind11;
using FLContext = mindspore::fl::FLContext;
using FederatedJob = mindspore::fl::FederatedJob;
using StartFLJobKernelMod = mindspore::fl::worker::kernel::StartFLJobKernelMod;

namespace mindspore {
namespace fl {
// Interface with python
PYBIND11_MODULE(_mindspore_federated, m) {
  (void)py::class_<FederatedJob, std::shared_ptr<FederatedJob>>(m, "Federated_")
    .def_static("start_federated_job", &FederatedJob::StartFederatedJob)
    .def_static("start_fl_job", &FederatedJob::StartFLJob)
    .def_static("update_and_get_model", &FederatedJob::UpdateAndGetModel);

  (void)py::class_<FLContext, std::shared_ptr<FLContext>>(m, "FLContext")
    .def_static("get_instance", &FLContext::instance, "Get fl context instance.")
    .def("reset", &FLContext::Reset, "Reset fl context attributes.")
    .def("is_worker", &FLContext::is_worker, "Get whether the role of this process is Worker.")
    .def("is_server", &FLContext::is_server, "Get whether the role of this process is Server.")
    .def("is_scheduler", &FLContext::is_scheduler, "Get whether the role of this process is Scheduler.")
    .def("set_server_mode", &FLContext::set_server_mode, "Set server mode.")
    .def("server_mode", &FLContext::server_mode, "Get server mode.")
    .def("set_ms_role", &FLContext::set_ms_role, "Set role for this process.")
    .def("ms_role", &FLContext::ms_role, "Get role for this process.")
    .def("set_worker_num", &FLContext::set_worker_num, "Set worker number.")
    .def("worker_num", &FLContext::initial_worker_num, "Get worker number.")
    .def("set_server_num", &FLContext::set_server_num, "Set server number.")
    .def("server_num", &FLContext::initial_server_num, "Get server number.")
    .def("set_scheduler_ip", &FLContext::set_scheduler_ip, "Set scheduler ip.")
    .def("scheduler_ip", &FLContext::scheduler_ip, "Get scheduler ip.")
    .def("set_scheduler_port", &FLContext::set_scheduler_port, "Set scheduler port.")
    .def("scheduler_port", &FLContext::scheduler_port, "Get scheduler port.")
    .def("set_fl_server_port", &FLContext::set_fl_server_port, "Set federated learning server port.")
    .def("fl_server_port", &FLContext::fl_server_port, "Get federated learning server port.")
    .def("set_fl_client_enable", &FLContext::set_fl_client_enable, "Set federated learning client.")
    .def("fl_client_enable", &FLContext::fl_client_enable, "Get federated learning client.")
    .def("set_start_fl_job_threshold", &FLContext::set_start_fl_job_threshold,
         "Set threshold count for startFLJob round.")
    .def("start_fl_job_threshold", &FLContext::start_fl_job_threshold, "Get threshold count for startFLJob round.")
    .def("set_start_fl_job_time_window", &FLContext::set_start_fl_job_time_window,
         "Set time window for startFLJob round.")
    .def("start_fl_job_time_window", &FLContext::start_fl_job_time_window, "Get time window for startFLJob round.")
    .def("set_update_model_ratio", &FLContext::set_update_model_ratio,
         "Set threshold count ratio for updateModel round.")
    .def("update_model_ratio", &FLContext::update_model_ratio, "Get threshold count ratio for updateModel round.")
    .def("set_update_model_time_window", &FLContext::set_update_model_time_window,
         "Set time window for updateModel round.")
    .def("update_model_time_window", &FLContext::update_model_time_window, "Get time window for updateModel round.")
    .def("set_share_secrets_ratio", &FLContext::set_share_secrets_ratio,
         "Set threshold count ratio for share secrets round.")
    .def("share_secrets_ratio", &FLContext::share_secrets_ratio, "Get threshold count ratio for share secrets round.")
    .def("set_cipher_time_window", &FLContext::set_cipher_time_window, "Set time window for each cipher round.")
    .def("cipher_time_window", &FLContext::cipher_time_window, "Get time window for cipher rounds.")
    .def("set_reconstruct_secrets_threshold", &FLContext::set_reconstruct_secrets_threshold,
         "Set threshold count for reconstruct secrets round.")
    .def("reconstruct_secrets_threshold", &FLContext::reconstruct_secrets_threshold,
         "Get threshold count for reconstruct secrets round.")
    .def("set_fl_name", &FLContext::set_fl_name, "Set federated learning name.")
    .def("fl_name", &FLContext::fl_name, "Get federated learning name.")
    .def("set_fl_iteration_num", &FLContext::set_fl_iteration_num, "Set federated learning iteration number.")
    .def("fl_iteration_num", &FLContext::fl_iteration_num, "Get federated learning iteration number.")
    .def("set_client_epoch_num", &FLContext::set_client_epoch_num, "Set federated learning client epoch number.")
    .def("client_epoch_num", &FLContext::client_epoch_num, "Get federated learning client epoch number.")
    .def("set_client_batch_size", &FLContext::set_client_batch_size, "Set federated learning client batch size.")
    .def("client_batch_size", &FLContext::client_batch_size, "Get federated learning client batch size.")
    .def("set_client_learning_rate", &FLContext::set_client_learning_rate,
         "Set federated learning client learning rate.")
    .def("client_learning_rate", &FLContext::client_learning_rate,
         "Get worker's standalone training step number before communicating with server.")
    .def("set_worker_step_num_per_iteration", &FLContext::set_worker_step_num_per_iteration,
         "Set worker's standalone training step number before communicating with server..")
    .def("worker_step_num_per_iteration", &FLContext::worker_step_num_per_iteration,
         "Get federated learning client learning rate.")
    .def("set_secure_aggregation", &FLContext::set_secure_aggregation,
         "Set federated learning client using secure aggregation.")
    .def("set_dp_eps", &FLContext::set_dp_eps, "Set dp epsilon for federated learning secure aggregation.")
    .def("dp_eps", &FLContext::dp_eps, "Get dp epsilon for federated learning secure aggregation.")
    .def("set_dp_delta", &FLContext::set_dp_delta, "Set dp delta for federated learning secure aggregation.")
    .def("dp_delta", &FLContext::dp_delta, "Get dp delta for federated learning secure aggregation.")
    .def("set_dp_norm_clip", &FLContext::set_dp_norm_clip,
         "Set dp norm clip for federated learning secure aggregation.")
    .def("dp_norm_clip", &FLContext::dp_norm_clip, "Get dp norm clip for federated learning secure aggregation.")
    .def("set_encrypt_type", &FLContext::set_encrypt_type,
         "Set encrypt type for federated learning secure aggregation.")
    .def("encrypt_type", &FLContext::encrypt_type, "Get encrypt type for federated learning secure aggregation.")
    .def("set_root_first_ca_path", &FLContext::set_root_first_ca_path, "Set root first ca path.")
    .def("root_first_ca_path", &FLContext::root_first_ca_path, "Get root first ca path.")
    .def("set_root_second_ca_path", &FLContext::set_root_second_ca_path, "Set root second ca path.")
    .def("root_second_ca_path", &FLContext::root_second_ca_path, "Get root second ca path.")
    .def("set_pki_verify", &FLContext::set_pki_verify, "Set pki verify.")
    .def("pki_verify", &FLContext::pki_verify, "Get pki verify.")
    .def("set_scheduler_manage_port", &FLContext::set_scheduler_manage_port,
         "Set scheduler manage port used to scale out/in.")
    .def("scheduler_manage_port", &FLContext::scheduler_manage_port, "Get scheduler manage port used to scale out/in.")
    .def("set_equip_crl_path", &FLContext::set_equip_crl_path, "Set root second crl path.")
    .def("set_replay_attack_time_diff", &FLContext::set_replay_attack_time_diff, "Set replay attack time diff.")
    .def("equip_crl_path", &FLContext::equip_crl_path, "Get root second crl path.")
    .def("replay_attack_time_diff", &FLContext::replay_attack_time_diff, "Get replay attack time diff.")
    .def("set_enable_ssl", &FLContext::set_enable_ssl, "Set PS SSL mode enabled or disabled.")
    .def("enable_ssl", &FLContext::enable_ssl, "Get PS SSL mode enabled or disabled.")
    .def("set_client_password", &FLContext::set_client_password, "Set the client password to decode the p12 file.")
    .def("client_password", &FLContext::client_password, "Get the client password to decode the p12 file.")
    .def("set_server_password", &FLContext::set_server_password, "Set the server password to decode the p12 file.")
    .def("server_password", &FLContext::server_password, "Get the server password to decode the p12 file.")
    .def("set_config_file_path", &FLContext::set_config_file_path,
         "Set configuration files required by the communication layer.")
    .def("config_file_path", &FLContext::config_file_path,
         "Get configuration files required by the communication layer.")
    .def("set_encrypt_type", &FLContext::set_encrypt_type,
         "Set encrypt type for federated learning secure aggregation.")
    .def("set_sign_k", &FLContext::set_sign_k, "Set sign k for federated learning SignDS.")
    .def("sign_k", &FLContext::sign_k, "Get sign k for federated learning SignDS.")
    .def("set_sign_eps", &FLContext::set_sign_eps, "Set sign eps for federated learning SignDS.")
    .def("sign_eps", &FLContext::sign_eps, "Get sign eps for federated learning SignDS.")
    .def("set_sign_thr_ratio", &FLContext::set_sign_thr_ratio, "Set sign thr ratio for federated learning SignDS.")
    .def("sign_thr_ratio", &FLContext::sign_thr_ratio, "Get sign thr ratio for federated learning SignDS.")
    .def("set_sign_global_lr", &FLContext::set_sign_global_lr, "Set sign global lr for federated learning SignDS.")
    .def("sign_global_lr", &FLContext::sign_global_lr, "Get sign global lr for federated learning SignDS.")
    .def("set_sign_dim_out", &FLContext::set_sign_dim_out, "Set sign dim out for federated learning SignDS.")
    .def("sign_dim_out", &FLContext::sign_dim_out, "Get sign dim out for federated learning SignDS.")
    .def("set_http_url_prefix", &FLContext::set_http_url_prefix, "Set http url prefix for http communication.")
    .def("http_url_prefix", &FLContext::http_url_prefix, "http url prefix for http communication.")
    .def("set_global_iteration_time_window", &FLContext::set_global_iteration_time_window,
         "Set global iteration time window.")
    .def("global_iteration_time_window", &FLContext::global_iteration_time_window, "Get global iteration time window.")
    .def("set_upload_compress_type", &FLContext::set_upload_compress_type, "Set upload compress type.")
    .def("upload_compress_type", &FLContext::upload_compress_type, "Get upload compress type.")
    .def("set_upload_sparse_rate", &FLContext::set_upload_sparse_rate, "Set upload sparse rate.")
    .def("upload_sparse_rate", &FLContext::upload_sparse_rate, "Get upload sparse rate.")
    .def("set_download_compress_type", &FLContext::set_download_compress_type, "Set download compress type.")
    .def("download_compress_type", &FLContext::download_compress_type, "Get download compress type.")
    .def("set_checkpoint_dir", &FLContext::set_checkpoint_dir, "Set server checkpoint directory.")
    .def("checkpoint_dir", &FLContext::checkpoint_dir, "Server checkpoint directory.")
    .def("set_instance_name", &FLContext::set_instance_name, "Set instance name.")
    .def("instance_name", &FLContext::instance_name, "Get instance name.")
    .def("set_participation_time_level", &FLContext::set_participation_time_level, "Set participation time level.")
    .def("participation_time_level", &FLContext::participation_time_level, "Get participation time level.")
    .def("set_continuous_failure_times", &FLContext::set_continuous_failure_times, "Set continuous failure times")
    .def("continuous_failure_times", &FLContext::continuous_failure_times, "Get continuous failure times.")
    .def("set_feature_maps", &FLContext::set_feature_maps, "Set Feature maps");
}
}  // namespace fl
}  // namespace mindspore
