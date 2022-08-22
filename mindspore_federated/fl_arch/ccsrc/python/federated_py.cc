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
#include "common/fl_context.h"
#include "python/federated_job.h"
#include "python/feature_py.h"
#include "worker/kernel/start_fl_job_kernel.h"
#include "vertical/vfl_context.h"
#include "vertical/python/vfederated_job.h"
#include "vertical/python/tensor_list_py.h"
#include "vertical/python/tensor_py.h"

namespace py = pybind11;
using FLContext = mindspore::fl::FLContext;
using VFLContext = mindspore::fl::VFLContext;
using FederatedJob = mindspore::fl::FederatedJob;
using VFederatedJob = mindspore::fl::VFederatedJob;
using StartFLJobKernelMod = mindspore::fl::worker::kernel::StartFLJobKernelMod;

namespace mindspore {
namespace fl {
// Interface with python
void InitVFLContext(const py::module &m) {
  (void)py::class_<VFLContext, std::shared_ptr<VFLContext>>(m, "VFLContext")
    .def_static("get_instance", &VFLContext::instance, "Get fl context instance.")
    .def("set_http_server_address", &VFLContext::set_http_server_address, "Set federated learning http server address.")
    .def("http_server_address", &VFLContext::http_server_address, "Get federated learning http server address.")
    .def("set_remote_server_address", &VFLContext::set_remote_server_address,
         "Set federated learning remote server address.")
    .def("remote_server_address", &VFLContext::remote_server_address, "Get federated learning remote server address.")
    .def("http_server_address", &VFLContext::http_server_address, "Get federated learning http server address.")
    .def("set_enable_ssl", &VFLContext::set_enable_ssl, "Set PS SSL mode enabled or disabled.")
    .def("enable_ssl", &VFLContext::enable_ssl, "Get PS SSL mode enabled or disabled.")
    .def("set_client_password", &VFLContext::set_client_password, "Set the client password to decode the p12 file.")
    .def("client_password", &VFLContext::client_password, "Get the client password to decode the p12 file.")
    .def("set_server_password", &VFLContext::set_server_password, "Set the server password to decode the p12 file.")
    .def("server_password", &VFLContext::server_password, "Get the server password to decode the p12 file.")
    .def("http_url_prefix", &VFLContext::http_url_prefix, "http url prefix for http communication.")
    .def("load_yaml_config", &VFLContext::LoadYamlConfig, "Load yaml config");
}

void InitTensorItemPy(const py::module &m) {
  (void)py::class_<TensorItemPy, std::shared_ptr<TensorItemPy>>(m, "TensorItem_")
    .def(py::init<>())
    .def("set_name", &TensorItemPy::set_name, "Set name.")
    .def("set_ref_key", &TensorItemPy::set_ref_key, "Get tensors.")
    .def("set_shape", &TensorItemPy::set_shape, "Set shape.")
    .def("set_dtype", &TensorItemPy::set_dtype, "Set dtype.")
    .def("set_data", &TensorItemPy::set_data, "Set data.")
    .def("name", &TensorItemPy::name, "Get name.")
    .def("ref_key", &TensorItemPy::ref_key, "Get ref_key.")
    .def("shape", &TensorItemPy::shape, "Get shape.")
    .def("dtype", &TensorItemPy::dtype, "Get dtype.")
    .def("data", &TensorItemPy::data, "Get data.");
}

void InitTensorListItemPy(const py::module &m) {
  (void)py::class_<TensorListItemPy, std::shared_ptr<TensorListItemPy>>(m, "TensorListItem_")
    .def(py::init<>())
    .def(py::init<const std::string &, const std::vector<TensorItemPy> &, const std::vector<TensorListItemPy> &>())
    .def("name", &TensorListItemPy::name, "Get tensor list name.")
    .def("tensors", &TensorListItemPy::tensors, "Get tensors.")
    .def("tensorListItems", &TensorListItemPy::tensorListItems, "Get tensorListItems.")
    .def("set_name", &TensorListItemPy::set_name, "Set name.")
    .def("add_tensor", &TensorListItemPy::add_tensor, "Add tensor.")
    .def("add_tensor_list_item", &TensorListItemPy::add_tensor_list_item, "Add tensor list item.");
}

// cppcheck-suppress syntaxError
PYBIND11_MODULE(_mindspore_federated, m) {
  (void)py::class_<FederatedJob, std::shared_ptr<FederatedJob>>(m, "Federated_")
    .def_static("start_federated_server", &FederatedJob::StartFederatedServer)
    .def_static("start_federated_scheduler", &FederatedJob::StartFederatedScheduler)
    .def_static("init_federated_worker", &FederatedJob::InitFederatedWorker)
    .def_static("stop_federated_worker", &FederatedJob::StopFederatedWorker)
    .def_static("start_fl_job", &FederatedJob::StartFLJob)
    .def_static("update_and_get_model", &FederatedJob::UpdateAndGetModel)
    .def_static("pull_weight", &FederatedJob::PullWeight)
    .def_static("push_weight", &FederatedJob::PushWeight)
    .def_static("push_metrics", &FederatedJob::PushMetrics);

  (void)py::class_<FLContext, std::shared_ptr<FLContext>>(m, "FLContext")
    .def_static("get_instance", &FLContext::instance, "Get fl context instance.")
    .def("reset", &FLContext::Reset, "Reset fl context attributes.")
    .def("is_worker", &FLContext::is_worker, "Get whether the role of this process is Worker.")
    .def("is_server", &FLContext::is_server, "Get whether the role of this process is Server.")
    .def("is_scheduler", &FLContext::is_scheduler, "Get whether the role of this process is Scheduler.")
    .def("server_mode", &FLContext::server_mode, "Get server mode.")
    .def("ms_role", &FLContext::ms_role, "Get role for this process.")
    .def("set_http_server_address", &FLContext::set_http_server_address, "Set federated learning http server address.")
    .def("http_server_address", &FLContext::http_server_address, "Get federated learning http server address.")
    .def("set_tcp_server_ip", &FLContext::set_tcp_server_ip, "Set federated learning tcp server ip.")
    .def("tcp_server_ip", &FLContext::tcp_server_ip, "Get federated learning tcp server ip.")
    .def("start_fl_job_threshold", &FLContext::start_fl_job_threshold, "Get threshold count for startFLJob round.")
    .def("start_fl_job_time_window", &FLContext::start_fl_job_time_window, "Get time window for startFLJob round.")
    .def("update_model_ratio", &FLContext::update_model_ratio, "Get threshold count ratio for updateModel round.")
    .def("update_model_time_window", &FLContext::update_model_time_window, "Get time window for updateModel round.")
    .def("fl_name", &FLContext::fl_name, "Get federated learning name.")
    .def("fl_iteration_num", &FLContext::fl_iteration_num, "Get federated learning iteration number.")
    .def("client_epoch_num", &FLContext::client_epoch_num, "Get federated learning client epoch number.")
    .def("client_batch_size", &FLContext::client_batch_size, "Get federated learning client batch size.")
    .def("encrypt_type", &FLContext::encrypt_type, "Get encrypt type for federated learning secure aggregation.")
    .def("client_learning_rate", &FLContext::client_learning_rate,
         "Get worker's standalone training step number before communicating with server.")
    .def("set_secure_aggregation", &FLContext::set_secure_aggregation,
         "Set federated learning client using secure aggregation.")
    .def("set_scheduler_manage_address", &FLContext::set_scheduler_manage_address, "Set scheduler manage http address.")
    .def("scheduler_manage_address", &FLContext::scheduler_manage_address, "Get scheduler manage http address.")
    .def("set_enable_ssl", &FLContext::set_enable_ssl, "Set PS SSL mode enabled or disabled.")
    .def("enable_ssl", &FLContext::enable_ssl, "Get PS SSL mode enabled or disabled.")
    .def("set_client_password", &FLContext::set_client_password, "Set the client password to decode the p12 file.")
    .def("client_password", &FLContext::client_password, "Get the client password to decode the p12 file.")
    .def("set_server_password", &FLContext::set_server_password, "Set the server password to decode the p12 file.")
    .def("server_password", &FLContext::server_password, "Get the server password to decode the p12 file.")
    .def("http_url_prefix", &FLContext::http_url_prefix, "http url prefix for http communication.")
    .def("set_global_iteration_time_window", &FLContext::set_global_iteration_time_window,
         "Set global iteration time window.")
    .def("global_iteration_time_window", &FLContext::global_iteration_time_window, "Get global iteration time window.")
    .def("set_checkpoint_dir", &FLContext::set_checkpoint_dir, "Set server checkpoint directory.")
    .def("checkpoint_dir", &FLContext::checkpoint_dir, "Server checkpoint directory.")
    .def("set_instance_name", &FLContext::set_instance_name, "Set instance name.")
    .def("instance_name", &FLContext::instance_name, "Get instance name.")
    .def("participation_time_level", &FLContext::participation_time_level, "Get participation time level.")
    .def("continuous_failure_times", &FLContext::continuous_failure_times, "Get continuous failure times.")
    .def("load_yaml_config", &FLContext::LoadYamlConfig, "Load yaml config");

  (void)py::class_<FeatureItemPy, std::shared_ptr<FeatureItemPy>>(m, "FeatureItem_")
    .def(py::init<const std::string &, const py::array &, const std::vector<size_t> &, const std::string &, bool>())
    .def_property_readonly("feature_name", &FeatureItemPy::feature_name, "Get feature name.")
    .def_property_readonly("shape", &FeatureItemPy::shape, "Get shape.")
    .def_property_readonly("data", &FeatureItemPy::data, "Get data.")
    .def_property_readonly("require_aggr", &FeatureItemPy::require_aggr, "Whether requires aggr.");

  // for zero copy
  (void)py::class_<ModelItem, std::shared_ptr<ModelItem>>(m, "ModelItem_");

  (void)py::class_<yaml::YamlConfigItem>(m, "YamlConfigItem_")
    .def(py::init<>())
    .def(
      "set_dict", [](yaml::YamlConfigItem &item) { item.type = yaml::kYamlDict; }, "Add feature.")
    .def("set_bool_val",
         [](yaml::YamlConfigItem &item, bool val) {
           item.type = yaml::kYamlBool;
           item.bool_val = val;
         })
    .def("set_int_val",
         [](yaml::YamlConfigItem &item, int64_t val) {
           item.type = yaml::kYamlInt;
           item.int_val = val;
         })
    .def("set_float_val",
         [](yaml::YamlConfigItem &item, float val) {
           item.type = yaml::kYamlFloat;
           item.float_val = val;
         })
    .def("set_str_val", [](yaml::YamlConfigItem &item, const std::string &val) {
      item.type = yaml::kYamlStr;
      item.str_val = val;
    });

  (void)py::class_<VFederatedJob, std::shared_ptr<VFederatedJob>>(m, "VFederated_")
    .def_static("start_trainer_communicator", &VFederatedJob::StartTrainerCommunicator)
    .def_static("send", &VFederatedJob::Send)
    .def_static("receive", &VFederatedJob::Receive);

  InitVFLContext(m);
  InitTensorItemPy(m);
  InitTensorListItemPy(m);
}
}  // namespace fl
}  // namespace mindspore
