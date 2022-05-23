/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "server/server.h"
#include <memory>
#include <string>
#include <csignal>
#include "armour/secure_protocol/secret_sharing.h"
#include "armour/cipher/cipher_init.h"
#include "server/round.h"
#include "server/model_store.h"
#include "server/iteration.h"
#include "server/collective_ops_impl.h"
#include "server/cert_verify.h"
#include "common/core/comm_util.h"
#include "common/utils/ms_exception.h"
#include "distributed_cache/distributed_cache.h"
#include "distributed_cache/client_infos.h"
#include "distributed_cache/instance_context.h"
#include "distributed_cache/hyper_params.h"
#include "distributed_cache/server.h"
#include "distributed_cache/timer.h"
#include "distributed_cache/counter.h"
#include "python/feature_py.h"

namespace mindspore {
namespace fl {
namespace server {
// The handler to capture the signal of SIGTERM. Normally this signal is triggered by cloud cluster manager like K8S.
namespace {
int g_signal = 0;
}
void SignalHandler(int signal) {
  if (g_signal == 0) {
    g_signal = signal;
    Server::GetInstance().SetStopFlag();
  }
}

Server &Server::GetInstance() {
  static Server instance;
  return instance;
}

Server::~Server() = default;

void Server::SetStopFlag() { stop_flag_ = true; }

void Server::Run(const std::vector<InputWeight> &feature_map, const FlCallback &fl_callback) {
  (void)signal(SIGTERM, SignalHandler);
  (void)signal(SIGINT, SignalHandler);
  fl_callback_ = fl_callback;
  try {
    InitServer();
    InitAndLoadDistributedCache();
    // Lock to prevent multiple servers from registering at the same time
    LockCache();
    // Visit all other servers, and ensure that the servers are reachable to each other.
    PingOtherServers();

    InitServerContext();
    InitPkiCertificate();
    InitCluster();
    InitExecutor(feature_map);
    InitIteration();
    StartCommunicator();

    // register server to distributed cache, and unlock to allow other servers to continue registering.
    RegisterServer();
    UnlockCache();
    MS_LOG(INFO) << "Server started successfully.";
    CallServerStartedCallback();
    RunMainProcess();
  } catch (const std::exception &e) {
    MS_LOG_WARNING << "Catch exception and begin exit, exception: " << e.what();
    UnlockCache();
    MsException::Instance().SetException();
  }
  if (g_signal != 0) {
    MS_LOG_INFO << "Receive signal message " << g_signal << " and begin exit";
  }
  CallServerStoppedCallback();
  Stop();
  MsException::Instance().CheckException();
}

void Server::InitRoundConfigs() {
  // Update model threshold is a certain ratio of start_fl_job threshold.
  // update_model_threshold = start_fl_job_threshold * update_model_ratio.
  uint64_t start_fl_job_threshold = FLContext::instance()->start_fl_job_threshold();
  float update_model_ratio = FLContext::instance()->update_model_ratio();
  uint64_t update_model_threshold = static_cast<uint64_t>(std::ceil(start_fl_job_threshold * update_model_ratio));
  uint64_t start_fl_job_time_window = FLContext::instance()->start_fl_job_time_window();
  uint64_t update_model_time_window = FLContext::instance()->update_model_time_window();

  std::vector<RoundConfig> rounds_config = {
    {"startFLJob", true, start_fl_job_time_window, true, start_fl_job_threshold},
    {"updateModel", true, update_model_time_window, true, update_model_threshold, true},
    {"getModel"},
    {"pullWeight"},
    {"pushWeight", false, 3000, false, 1},
    {"pushMetrics", false, 3000, true, 1}};

  auto encrypt_config = FLContext::instance()->encrypt_config();
  float share_secrets_ratio = encrypt_config.share_secrets_ratio;
  uint64_t cipher_time_window = encrypt_config.cipher_time_window;
  cipher_config_.share_secrets_ratio = share_secrets_ratio;
  cipher_config_.cipher_time_window = cipher_time_window;
  std::string encrypt_type = FLContext::instance()->encrypt_type();

  cipher_config_.minimum_secret_shares_for_reconstruct = encrypt_config.reconstruct_secrets_threshold;
  cipher_config_.minimum_clients_for_reconstruct = cipher_config_.minimum_secret_shares_for_reconstruct + 1;
  cipher_config_.exchange_keys_threshold =
    std::max(static_cast<uint64_t>(std::ceil(start_fl_job_threshold * share_secrets_ratio)), update_model_threshold);
  cipher_config_.get_keys_threshold =
    std::max(static_cast<uint64_t>(std::ceil(cipher_config_.exchange_keys_threshold * share_secrets_ratio)),
             update_model_threshold);
  cipher_config_.share_secrets_threshold = std::max(
    static_cast<uint64_t>(std::ceil(cipher_config_.get_keys_threshold * share_secrets_ratio)), update_model_threshold);
  cipher_config_.get_secrets_threshold =
    std::max(static_cast<uint64_t>(std::ceil(cipher_config_.share_secrets_threshold * share_secrets_ratio)),
             update_model_threshold);
  cipher_config_.get_client_list_threshold =
    std::max(static_cast<uint64_t>(std::ceil(update_model_threshold * share_secrets_ratio)),
             cipher_config_.minimum_clients_for_reconstruct);
  cipher_config_.push_list_sign_threshold =
    std::max(static_cast<uint64_t>(std::ceil(cipher_config_.get_client_list_threshold * share_secrets_ratio)),
             cipher_config_.minimum_clients_for_reconstruct);
  cipher_config_.get_list_sign_threshold =
    std::max(static_cast<uint64_t>(std::ceil(cipher_config_.push_list_sign_threshold * share_secrets_ratio)),
             cipher_config_.minimum_clients_for_reconstruct);
  if (encrypt_type == kPWEncryptType) {
    MS_LOG(INFO) << "Add secure aggregation rounds.";
    rounds_config.push_back({"exchangeKeys", true, cipher_time_window, true, cipher_config_.exchange_keys_threshold});
    rounds_config.push_back({"getKeys", true, cipher_time_window, true, cipher_config_.get_keys_threshold});
    rounds_config.push_back({"shareSecrets", true, cipher_time_window, true, cipher_config_.share_secrets_threshold});
    rounds_config.push_back({"getSecrets", true, cipher_time_window, true, cipher_config_.get_secrets_threshold});
    rounds_config.push_back(
      {"getClientList", true, cipher_time_window, true, cipher_config_.get_client_list_threshold});
    rounds_config.push_back(
      {"reconstructSecrets", true, cipher_time_window, true, cipher_config_.minimum_clients_for_reconstruct});
    if (FLContext::instance()->pki_verify()) {
      rounds_config.push_back(
        {"pushListSign", true, cipher_time_window, true, cipher_config_.push_list_sign_threshold});
      rounds_config.push_back({"getListSign", true, cipher_time_window, true, cipher_config_.get_list_sign_threshold});
    }
  }
  if (encrypt_type == kStablePWEncryptType) {
    MS_LOG(INFO) << "Add stable secure aggregation rounds.";
    rounds_config.push_back({"exchangeKeys", true, cipher_time_window, true, cipher_config_.exchange_keys_threshold});
    rounds_config.push_back({"getKeys", true, cipher_time_window, true, cipher_config_.get_keys_threshold});
  }
  MS_LOG(INFO) << "Initializing cipher:";
  MS_LOG(INFO) << " exchange_keys_threshold: " << cipher_config_.exchange_keys_threshold
               << " get_keys_threshold: " << cipher_config_.get_keys_threshold
               << " share_secrets_threshold: " << cipher_config_.share_secrets_threshold;
  MS_LOG(INFO) << " get_secrets_threshold: " << cipher_config_.get_secrets_threshold
               << " get_client_list_threshold: " << cipher_config_.get_client_list_threshold
               << " push_list_sign_threshold: " << cipher_config_.push_list_sign_threshold
               << " get_list_sign_threshold: " << cipher_config_.get_list_sign_threshold
               << " minimum_clients_for_reconstruct: " << cipher_config_.minimum_clients_for_reconstruct
               << " minimum_secret_shares_for_reconstruct: " << cipher_config_.minimum_secret_shares_for_reconstruct
               << " cipher_time_window: " << cipher_config_.cipher_time_window;
  rounds_config_ = rounds_config;
}

void Server::RunMainProcess() {
  MS_EXCEPTION_IF_NULL(server_node_);
  MS_LOG_INFO << "Begin run main process, start iteration: " << cache::InstanceContext::Instance().iteration_num()
              << ", instance name: " << cache::InstanceContext::Instance().instance_name() << ", instance state: "
              << cache::GetInstanceStateStr(cache::InstanceContext::Instance().instance_state());
  Iteration::GetInstance().StartThreadToRecordDataRate();
  cache::IterationTaskThread::Instance().Start();
  while (!HasStopped()) {
    RunMainProcessInner();
    constexpr int default_sync_duration_ms = 1000;  // 1000ms
    std::this_thread::sleep_for(std::chrono::milliseconds(default_sync_duration_ms));
  }
  cache::IterationTaskThread::Instance().Stop();
  Iteration::GetInstance().Stop();
  MS_LOG_INFO << "End run main process";
}

void Server::CallIterationEndCallback() {
  try {
    auto callback = fl_callback_.on_iteration_end;
    if (!callback) {
      return;
    }
    callback();
  } catch (const std::exception &e) {
    MS_LOG_WARNING << "Catch exception when invoke callback on iteration end: " << e.what();
  }
}

void Server::CallServerStartedCallback() {
  try {
    auto callback = fl_callback_.after_started;
    if (!callback) {
      return;
    }
    callback();
  } catch (const std::exception &e) {
    MS_LOG_WARNING << "Catch exception when invoke callback after stated: " << e.what();
  }
}

void Server::CallServerStoppedCallback() {
  try {
    auto callback = fl_callback_.before_stopped;
    if (!callback) {
      return;
    }
    callback();
  } catch (const std::exception &e) {
    MS_LOG_WARNING << "Catch exception when invoke callback before stopped: " << e.what();
  }
}

void Server::RunMainProcessInner() {
  cache::CacheStatus cache_ret;
  if (cache::DistributedCacheLoader::Instance().HasInvalid()) {
    cache_ret = cache::DistributedCacheLoader::Instance().RetryConnect();
    if (!cache_ret.IsSuccess()) {
      constexpr int64_t log_interval = 15000;  // 15s
      static int64_t last_retry_timestamp_ms = 0;
      int64_t current_timestamp_ms = CURRENT_TIME_MILLI.count();
      if (current_timestamp_ms - last_retry_timestamp_ms >= log_interval) {
        last_retry_timestamp_ms = current_timestamp_ms;
        MS_LOG_ERROR << "Failed to reconnect to distributed cache: " << cache_ret.GetDetail();
      }
      cache::DistributedCacheLoader::Instance().set_available(false);
      return;
    }
    MS_LOG_INFO << "Success to reconnect to distributed cache";
  }
  cache::DistributedCacheLoader::Instance().set_available(true);
  // sync with cache, heartbeat and update infos of all servers.
  cache_ret = cache::Server::Instance().Sync();
  if (!cache_ret.IsSuccess()) {
    MS_LOG_WARNING << "Sync server info with distributed cache failed";
    return;
  }
  // sync with cache, get info about iteration and instance
  auto &instance_context = cache::InstanceContext::Instance();
  cache_ret = instance_context.Sync();
  if (!cache_ret.IsSuccess()) {  // failed to sync with cache, retry later
    MS_LOG_WARNING << "Sync instance context with distributed cache failed";
    return;
  }
  // sync with cache, trigger counter event: first/last count.
  // result: start/stop timer, AllReduce
  cache::Counter::Instance().Sync();
  // sync with cache, trigger timer event: start, stop, timeout.
  // timeout result: move next iteration
  cache::Timer::Instance().Sync();
  // if hyper params info does not exist in the cache, sync local info to the cache
  cache::HyperParams::Instance().SyncPeriod();
  auto instance_event = instance_context.GetInstanceEventType();
  if (instance_event == cache::kInstanceEventNone) {
    if (cache::Counter::Instance().HasServerExit(kUpdateModelKernel)) {
      auto curr_iter_num = cache::InstanceContext::Instance().iteration_num();
      std::string reason =
        "Server that processed updateModel requests exited, current iteration: " + std::to_string(curr_iter_num);
      MS_LOG_WARNING << reason;
      Executor::GetInstance().FinishIteration(false, reason);
      instance_event = instance_context.GetInstanceEventType();
    }
    if (instance_event == cache::kInstanceEventNone) {
      return;
    }
  }
  if (instance_event != cache::kInstanceEventNewInstance &&
      instance_context.instance_state() != cache::InstanceState::kStateRunning) {
    return;
  }
  auto event_str = instance_event == cache::kInstanceEventNewIteration ? "EventNewIteration" : "EventNewInstance";
  MS_LOG_INFO << "Start handle instance event " << event_str << ", cur iteration: " << instance_context.iteration_num()
              << ", cur instance name: " << instance_context.instance_name();
  // Pause receiving client messages and events
  instance_context.SetSafeMode(true);
  auto &iteration_instance = Iteration::GetInstance();
  // all client request should be finished
  iteration_instance.WaitAllRoundsFinish();
  // Counter and timer handling including AllReduce should be finished
  cache::IterationTaskThread::Instance().WaitAllTaskFinish();
  // Save model for current iteration
  iteration_instance.SaveModel();
  // Summary for current iteration
  iteration_instance.SummaryOnIterationFinish([this]() { CallIterationEndCallback(); });
  // For new iteration
  iteration_instance.Reset();
  server_node_->OnIterationUpdate();
  Executor::GetInstance().ResetAggregationStatus();
  // update local cache, iteration num or instance name will be updated
  instance_context.HandleInstanceEvent();
  // For new instance, update local counter and timer, and the new configs will be synced to the cache of new instance.
  // The caches of the old and new instance are isolated.
  if (instance_event == cache::kInstanceEventNewInstance) {
    iteration_instance.StartNewInstance();
  }
  // Resume receiving client messages and events
  instance_context.SetSafeMode(false);
  MS_LOG_INFO << "End handle instance event " << event_str << ", cur iteration: " << instance_context.iteration_num()
              << ", cur instance name: " << instance_context.instance_name();
}

void Server::Stop() {
  if (has_stopped_) {
    return;
  }
  if (server_node_) {
    server_node_->Stop();
  }
  cache::Server::Instance().Stop();
  has_stopped_ = true;
}

void Server::InitPkiCertificate() {
  auto &client_verify_config = FLContext::instance()->client_verify_config();
  if (client_verify_config.pki_verify) {
    root_first_ca_path_ = client_verify_config.root_first_ca_path;
    root_second_ca_path_ = client_verify_config.root_second_ca_path;
    equip_crl_path_ = client_verify_config.equip_crl_path;
    replay_attack_time_diff_ = client_verify_config.replay_attack_time_diff;

    bool ret = CertVerify::initRootCertAndCRL(root_first_ca_path_, root_second_ca_path_, equip_crl_path_,
                                              replay_attack_time_diff_);
    if (!ret) {
      MS_LOG(EXCEPTION) << "init root cert and crl failed.";
    }
    return;
  }
}

void Server::InitServerContext() { FLContext::instance()->GenerateResetterRound(); }

void Server::InitAndLoadDistributedCache() {
  MS_EXCEPTION_IF_NULL(server_node_);
  auto config = FLContext::instance()->distributed_cache_config();
  if (config.address.empty()) {
    MS_LOG(EXCEPTION) << "Distributed cache address cannot be empty.";
  }
  if (!cache::DistributedCacheLoader::Instance().InitCacheImpl(config)) {
    MS_LOG(EXCEPTION) << "Link to distributed cache failed, distributed cache address: " << config.address
                      << ", enable ssl: " << config.enable_ssl;
  }
  auto fl_name = FLContext::instance()->fl_name();
  auto cache_ret = cache::InstanceContext::Instance().InitAndSync(fl_name);
  if (!cache_ret.IsSuccess()) {
    MS_LOG(EXCEPTION) << "Sync instance info with distributed cache failed, distributed_cache_address: "
                      << config.address;
  }
  // sync hyper params
  cache_ret = cache::HyperParams::Instance().InitAndSync();
  if (!cache_ret.IsSuccess()) {
    MS_LOG(EXCEPTION) << "Sync hyper params with distributed cache failed, distributed_cache_address: "
                      << config.address;
  }
  auto node_id = server_node_->node_id();
  auto tcp_address = server_node_->tcp_address();
  cache::Server::Instance().Init(node_id, tcp_address);
}

void Server::PingOtherServers() {
  // ping all other servers
  if (!server_node_->ServerPingPong()) {
    MS_LOG(EXCEPTION) << "Failed to access all other servers";
  }
}

void Server::RegisterServer() {
  auto status = cache::Server::Instance().Register();
  if (!status.IsSuccess()) {
    MS_LOG_EXCEPTION << "Failed to register server " << cache::Server::Instance().node_id() << " to distributed cache";
    return;
  }
  MS_LOG_INFO << "Success to register server " << cache::Server::Instance().node_id() << " to distributed cache";
}

void Server::LockCache() {
  if (!cache::Server::Instance().LockCache()) {
    MS_LOG_EXCEPTION << "Failed to lock server";
  }
}

void Server::UnlockCache() { cache::Server::Instance().UnlockCache(); }

void Server::InitServer() {
  server_node_ = std::make_shared<ServerNode>();
  MS_EXCEPTION_IF_NULL(server_node_);
  std::string tcp_server_ip = FLContext::instance()->tcp_server_ip();
  uint16_t server_port = 0;
  server_node_->InitializeBeforeCache(tcp_server_ip, server_port);
}

void Server::InitCluster() {
  if (!InitCommunicatorWithServer()) {
    MS_LOG(EXCEPTION) << "Initializing cross-server communicator failed.";
  }
  if (!InitCommunicatorWithWorker()) {
    MS_LOG(EXCEPTION) << "Initializing worker-server communicator failed.";
  }
}

bool Server::InitCommunicatorWithServer() {
  MS_EXCEPTION_IF_NULL(server_node_);
  communicator_with_server_ = server_node_->GetOrCreateTcpComm();

  MS_EXCEPTION_IF_NULL(communicator_with_server_);
  return true;
}

bool Server::InitCommunicatorWithWorker() {
  MS_EXCEPTION_IF_NULL(server_node_);
  MS_EXCEPTION_IF_NULL(communicator_with_server_);
  auto tcp_comm = communicator_with_server_;
  MS_EXCEPTION_IF_NULL(tcp_comm);
  communicators_with_worker_.push_back(tcp_comm);

  auto http_server_address = FLContext::instance()->http_server_address();
  std::string server_ip;
  uint32_t http_port = 0;
  if (!CommUtil::SplitIpAddress(http_server_address, &server_ip, &http_port)) {
    MS_LOG_EXCEPTION << "The format of http server address '" << http_server_address << "' is invalid";
  }
  auto http_comm = server_node_->GetOrCreateHttpComm(server_ip, http_port);
  MS_EXCEPTION_IF_NULL(http_comm);
  communicators_with_worker_.push_back(http_comm);
  return true;
}

void Server::InitIteration() {
  MS_EXCEPTION_IF_NULL(server_node_);
  InitRoundConfigs();
  std::string encrypt_type = FLContext::instance()->encrypt_type();
  if (encrypt_type != kNotEncryptType) {
    InitCipher();
    MS_LOG(INFO) << "Parameters for secure aggregation have been initiated.";
  }
  Iteration::GetInstance().InitIteration(server_node_, rounds_config_, communicators_with_worker_);
}

void Server::InitCipher() {
  cipher_init_ = &armour::CipherInit::GetInstance();

  int cipher_t = SizeToInt(cipher_config_.minimum_secret_shares_for_reconstruct);
  unsigned char cipher_p[SECRET_MAX_LEN] = {0};
  const int cipher_g = 1;

  armour::CipherPublicPara param;
  param.g = cipher_g;
  param.t = cipher_t;
  int ret = memcpy_s(param.p, SECRET_MAX_LEN, cipher_p, sizeof(cipher_p));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Memcpy_s error, errorno" << ret;
  }
  auto encrypt_config = FLContext::instance()->encrypt_config();
  param.dp_delta = encrypt_config.dp_delta;
  param.dp_eps = encrypt_config.dp_eps;
  param.dp_norm_clip = encrypt_config.dp_norm_clip;
  param.encrypt_type = encrypt_config.encrypt_type;
  param.sign_k = encrypt_config.sign_k;
  param.sign_eps = encrypt_config.sign_eps;
  param.sign_thr_ratio = encrypt_config.sign_thr_ratio;
  param.sign_global_lr = encrypt_config.sign_global_lr;
  param.sign_dim_out = static_cast<int>(encrypt_config.sign_dim_out);

  BIGNUM *prim = BN_new();
  if (prim == NULL) {
    MS_LOG(EXCEPTION) << "new bn failed.";
    ret = -1;
  } else {
    ret = armour::GetPrime(prim);
  }
  if (ret == 0) {
    (void)BN_bn2bin(prim, reinterpret_cast<uint8_t *>(param.prime));
  } else {
    MS_LOG(EXCEPTION) << "Get prime failed.";
  }
  if (prim != NULL) {
    BN_clear_free(prim);
  }
  if (!cipher_init_->Init(param, 0, cipher_config_)) {
    MS_LOG(EXCEPTION) << "cipher init fail.";
  }
}

void Server::InitExecutor(const std::vector<InputWeight> &init_feature_map) {
  auto status = SyncAndInitModel(init_feature_map);
  if (!status.IsSuccess()) {
    MS_LOG_EXCEPTION << "Load model failed: " << status.StatusMessage();
  }
  // reset weight memory to 0 after get model
  Executor::GetInstance().ResetAggregationStatus();
}

FlStatus Server::SyncAndInitModel(const std::vector<InputWeight> &init_feature_map) {
  if (server_node_ == nullptr) {
    auto reason = "server_node_ cannot be nullptr";
    MS_LOG_ERROR << reason;
    return {kFlFailed, reason};
  }
  // new_iteration_num is the iteration to be updated
  auto updated_iteration = cache::InstanceContext::Instance().new_iteration_num();
  if (updated_iteration <= 0) {
    auto reason = "Invalid iteration number: " + std::to_string(updated_iteration);
    return {kFlFailed, reason};
  }
  auto model_latest_iteration = updated_iteration - 1;
  VectorPtr output = nullptr;
  if (server_node_->GetModelWeight(model_latest_iteration, &output)) {
    ProtoModel proto_model;
    if (!proto_model.ParseFromArray(output->data(), output->size())) {
      auto reason = "Failed to store model synced from other servers";
      return {kFlFailed, reason};
    }
    std::vector<InputWeight> feature_map;
    for (auto &proto_weight : proto_model.weights()) {
      InputWeight weight;
      weight.name = proto_weight.name();
      weight.type = proto_weight.type();
      weight.size = proto_weight.data().size();
      weight.data = proto_weight.data().data();
      weight.requires_aggr = proto_weight.requires_aggr();
      auto &proto_shape = proto_weight.shape();
      std::copy(proto_shape.begin(), proto_shape.end(), std::back_inserter(weight.shape));
      feature_map.push_back(weight);
    }
    Executor::GetInstance().Initialize(feature_map, server_node_);
    MS_LOG_INFO << "Load model success: The model synced from other servers is used as the model of iteration "
                << model_latest_iteration;
  } else {
    Executor::GetInstance().Initialize(init_feature_map, server_node_);
    MS_LOG_INFO << "Load model success: The initial local model is used as the model of iteration "
                << model_latest_iteration;
  }
  return kFlSuccess;
}

void Server::StartCommunicator() {
  MS_EXCEPTION_IF_NULL(server_node_);
  server_node_->Start();
  CollectiveOpsImpl::GetInstance().Initialize(server_node_);
}

void Server::BroadcastModelWeight(const std::string &proto_model,
                                  const std::map<std::string, std::string> &broadcast_server_map) {
  if (server_node_ == nullptr) {
    MS_LOG_ERROR << "server_node_ cannot be nullptr";
    return;
  }
  server_node_->BroadcastModelWeight(proto_model, broadcast_server_map);
}

bool Server::PullWeight(const uint8_t *req_data, size_t len, VectorPtr *output) {
  if (server_node_ == nullptr) {
    MS_LOG_ERROR << "server_node_ cannot be nullptr";
    return false;
  }
  return server_node_->PullWeight(req_data, len, output);
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
