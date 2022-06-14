/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "scheduler/scheduler_node.h"
#include "distributed_cache/scheduler.h"
#include "distributed_cache/hyper_params.h"
#include "distributed_cache/instance_context.h"

namespace mindspore {
namespace fl {
SchedulerNode::~SchedulerNode() {
  MS_LOG(INFO) << "Stop scheduler node!";
  if (!Stop()) {
    MS_LOG(WARNING) << "Scheduler node stop failed.";
  }
}

bool SchedulerNode::Start(const uint32_t &timeout) {
  MS_LOG(INFO) << "[Scheduler start]: 1. Begin to start scheduler node!";
  auto scheduler_address = FLContext::instance()->scheduler_manage_address();
  std::string scheduler_ip;
  uint32_t scheduler_port = 0;
  if (!CommUtil::SplitIpAddress(scheduler_address, &scheduler_ip, &scheduler_port)) {
    MS_LOG_EXCEPTION << "Failed to start scheduler http server, invalid server address: " << scheduler_address;
  }
  MS_LOG(INFO) << "Start the restful scheduler http service, server address: " << scheduler_address;
  StartRestfulServer(scheduler_ip, scheduler_port, 1);
  return true;
}

void SchedulerNode::Initialize() {}

bool SchedulerNode::Stop() {
  MS_LOG(INFO) << "Stop scheduler node!";
  auto scheduler_address = FLContext::instance()->scheduler_manage_address();
  if (!scheduler_address.empty()) {
    MS_LOG(WARNING) << "Stop the restful scheduler http service, the ip is 127.0.0.1 "
                    << ", the port:" << scheduler_address;
    StopRestfulServer();
  }
  return true;
}

FlStatus SchedulerNode::GetClusterState(const std::string &fl_name, cache::InstanceState *state) {
  if (state == nullptr) {
    auto message = "Inner error";
    return FlStatus(kSystemError, message);
  }
  auto status = cache::Scheduler::Instance().GetAllClusterState(fl_name, state);
  if (status == cache::kCacheNetErr) {
    auto message = "Failed to access the cache server. Please retry later.";
    return FlStatus(kSystemError, message);
  }
  if (status == cache::kCacheNil) {
    auto message = "Cannot find cluster info for " + fl_name;
    return FlStatus(kSystemError, message);
  }
  if (!status.IsSuccess()) {
    auto message = "Failed to get cluster state because of some inner error.";
    return FlStatus(kSystemError, message);
  }
  return kFlSuccess;
}

FlStatus SchedulerNode::GetNodesInfoCommon(const std::string &fl_name, nlohmann::json *js) {
  MS_EXCEPTION_IF_NULL(js);
  std::map<std::string, std::string> server_map;
  auto cache_ret = cache::Scheduler::Instance().GetAllServersRealtime(fl_name, &server_map);
  if (cache_ret == cache::kCacheNetErr) {
    auto message = "Failed to access the cache server. Please retry later.";
    return FlStatus(kSystemError, message);
  }
  if (cache_ret == cache::kCacheNil) {
    auto message = "Cannot find cluster info for " + fl_name;
    return FlStatus(kSystemError, message);
  }
  if (!cache_ret.IsSuccess()) {
    auto message = "Failed to get nodes because of some inner error. Please retry later.";
    return FlStatus(kSystemError, message);
  }
  for (const auto &kvs : server_map) {
    std::unordered_map<std::string, std::string> res;
    res["node_id"] = kvs.first;
    res["tcp_address"] = kvs.second;
    res["role"] = "SERVER";
    (*js)["nodes"].push_back(res);
  }
  return FlStatus(kFlSuccess);
}

/*
 * The response body format.
 * {
 *    "message": "Get cluster state successful.",
 *    "cluster_state": "CLUSTER_READY"
 * }
 */
void SchedulerNode::ProcessGetClusterState(const std::shared_ptr<HttpMessageHandler> &resp) {
  MS_EXCEPTION_IF_NULL(resp);
  std::string fl_name = FLContext::instance()->fl_name();
  cache::InstanceState state;
  auto status = GetClusterState(fl_name, &state);
  if (!status.IsSuccess()) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }
  nlohmann::json js;
  // get nodes info
  status = GetNodesInfoCommon(fl_name, &js);
  if (!status.IsSuccess()) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }
  js["message"] = "Get cluster state successful.";
  js["code"] = kSuccessCode;
  switch (state) {
    case cache::kStateRunning:
      js["cluster_state"] = "CLUSTER_READY";
      break;
    case cache::kStateDisable:
      js["cluster_state"] = "CLUSTER_DISABLE";
      break;
    case cache::kStateFinish:
      js["cluster_state"] = "CLUSTER_FINISH";
      break;
    default:
      break;
  }
  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

void SchedulerNode::ProcessNewInstance(const std::shared_ptr<HttpMessageHandler> &resp) {
  MS_EXCEPTION_IF_NULL(resp);
  std::string fl_name = FLContext::instance()->fl_name();
  auto new_instance_name = cache::InstanceContext::CreateNewInstanceName();
  auto status = resp->ParsePostMessageToJson();
  if (!status.IsSuccess()) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }
  std::string hyper_params = resp->request_message().dump();
  std::string error_msg;
  std::string real_hyper_params;
  if (!cache::HyperParams::MergeHyperJsonConfig(fl_name, hyper_params, &error_msg, &real_hyper_params)) {
    resp->ErrorResponse(HTTP_BADREQUEST, "New instance failed: " + error_msg);
    return;
  }
  auto cache_ret = cache::Scheduler::Instance().OnNewInstance(fl_name, new_instance_name, real_hyper_params);
  if (cache_ret == cache::kCacheNetErr) {
    auto message = "Failed to access the cache server. Please retry later.";
    resp->ErrorResponse(HTTP_BADREQUEST, message);
    return;
  }
  if (!cache_ret.IsSuccess()) {
    auto message = "Failed to new instance because of some inner error. Please retry later.";
    resp->ErrorResponse(HTTP_BADREQUEST, message);
    return;
  }
  nlohmann::json js;
  js["message"] = "Start new instance successful.";
  js["code"] = kSuccessCode;

  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

void SchedulerNode::ProcessQueryInstance(const std::shared_ptr<HttpMessageHandler> &resp) {
  MS_EXCEPTION_IF_NULL(resp);

  FlStatus status = kFlSuccess;
  std::string fl_name = FLContext::instance()->fl_name();
  std::string hyper_params;
  auto cache_ret = cache::Scheduler::Instance().QueryInstance(fl_name, &hyper_params);
  if (cache_ret == cache::kCacheNetErr) {
    auto message = "Failed to access the cache server. Please retry later.";
    resp->ErrorResponse(HTTP_BADREQUEST, message);
    return;
  }
  if (cache_ret == cache::kCacheNil) {
    auto message = "Cannot find cluster info for " + fl_name;
    resp->ErrorResponse(HTTP_BADREQUEST, message);
    return;
  }
  if (!cache_ret.IsSuccess()) {
    auto message = "Failed to query instance because of some inner error. Please retry later.";
    resp->ErrorResponse(HTTP_BADREQUEST, message);
    return;
  }
  nlohmann::json js;
  // get nodes info
  status = GetNodesInfoCommon(fl_name, &js);
  if (!status.IsSuccess()) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }
  js["message"] = "Query Instance successful.";
  js["code"] = kSuccessCode;
  js["result"] = nlohmann::json::parse(hyper_params);

  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

void SchedulerNode::ProcessEnableFLS(const std::shared_ptr<HttpMessageHandler> &resp) {
  if (resp == nullptr) {
    return;
  }
  std::string fl_name = FLContext::instance()->fl_name();
  cache::InstanceState state;
  auto status = GetClusterState(fl_name, &state);
  if (!status.IsSuccess()) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }
  if (state == cache::kStateFinish) {
    auto message = "The instance is completed and cannot be enabled.";
    resp->ErrorResponse(HTTP_BADREQUEST, message);
    return;
  }
  if (state == cache::kStateRunning) {
    auto message = "The instance has already been enabled.";
    resp->ErrorResponse(HTTP_BADREQUEST, message);
    return;
  }
  auto cache_ret = cache::Scheduler::Instance().SetEnableState(fl_name, true);
  if (cache_ret == cache::kCacheNetErr) {
    auto message = "Failed to access the cache server. Please retry later.";
    resp->ErrorResponse(HTTP_BADREQUEST, message);
    return;
  }
  if (!cache_ret.IsSuccess()) {
    auto message = "Failed to enable cluster because of some inner error. Please retry later.";
    resp->ErrorResponse(HTTP_BADREQUEST, message);
    return;
  }
  nlohmann::json js;
  js["message"] = "start enabling FL-Server successful.";
  js["code"] = kSuccessCode;

  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

void SchedulerNode::ProcessDisableFLS(const std::shared_ptr<HttpMessageHandler> &resp) {
  if (resp == nullptr) {
    return;
  }
  std::string fl_name = FLContext::instance()->fl_name();
  cache::InstanceState state;
  auto status = GetClusterState(fl_name, &state);
  if (!status.IsSuccess()) {
    resp->ErrorResponse(HTTP_BADREQUEST, status);
    return;
  }
  if (state == cache::kStateFinish) {
    auto message = "The instance is completed and cannot be disabled.";
    resp->ErrorResponse(HTTP_BADREQUEST, message);
    return;
  }
  if (state == cache::kStateDisable) {
    auto message = "The instance has already been disabled.";
    resp->ErrorResponse(HTTP_BADREQUEST, message);
    return;
  }
  auto cache_ret = cache::Scheduler::Instance().SetEnableState(fl_name, false);
  if (cache_ret == cache::kCacheNetErr) {
    auto message = "Failed to access the cache server. Please retry later.";
    resp->ErrorResponse(HTTP_BADREQUEST, message);
    return;
  }
  if (!cache_ret.IsSuccess()) {
    auto message = "Failed to disable cluster because of some inner error. Please retry later.";
    resp->ErrorResponse(HTTP_BADREQUEST, message);
    return;
  }
  nlohmann::json js;
  js["message"] = "start disabling FL-Server successful.";
  js["code"] = kSuccessCode;

  resp->AddRespString(js.dump());
  resp->AddRespHeadParam("Content-Type", "application/json");

  resp->SetRespCode(HTTP_OK);
  resp->SendResponse();
}

void SchedulerNode::StartRestfulServer(const std::string &address, std::uint16_t port, size_t thread_num) {
  MS_LOG(INFO) << "Scheduler start https server.";
  http_server_ = std::make_shared<HttpServer>(address, port, thread_num);
  MS_EXCEPTION_IF_NULL(http_server_);

  OnRequestReceive cluster_state = std::bind(&SchedulerNode::ProcessGetClusterState, this, std::placeholders::_1);
  callbacks_["/state"] = cluster_state;
  http_server_->RegisterRoute("/state", &callbacks_["/state"]);

  OnRequestReceive new_instance = std::bind(&SchedulerNode::ProcessNewInstance, this, std::placeholders::_1);
  callbacks_["/newInstance"] = new_instance;
  (void)http_server_->RegisterRoute("/newInstance", &callbacks_["/newInstance"]);

  OnRequestReceive query_instance = std::bind(&SchedulerNode::ProcessQueryInstance, this, std::placeholders::_1);
  callbacks_["/queryInstance"] = query_instance;
  (void)http_server_->RegisterRoute("/queryInstance", &callbacks_["/queryInstance"]);

  OnRequestReceive enable_fls = std::bind(&SchedulerNode::ProcessEnableFLS, this, std::placeholders::_1);
  callbacks_["/enableFLS"] = enable_fls;
  (void)http_server_->RegisterRoute("/enableFLS", &callbacks_["/enableFLS"]);

  OnRequestReceive disable_fls = std::bind(&SchedulerNode::ProcessDisableFLS, this, std::placeholders::_1);
  callbacks_["/disableFLS"] = disable_fls;
  (void)http_server_->RegisterRoute("/disableFLS", &callbacks_["/disableFLS"]);

  if (!http_server_->Start()) {
    MS_LOG(EXCEPTION) << "The scheduler start http server failed, server address: " << address << ":" << port;
  }
}

void SchedulerNode::StopRestfulServer() {
  MS_LOG(INFO) << "Scheduler stop https server.";
  MS_ERROR_IF_NULL_WO_RET_VAL(http_server_);
  http_server_->Stop();
}
}  // namespace fl
}  // namespace mindspore
