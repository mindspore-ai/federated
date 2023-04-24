# yaml详细配置项

现有联邦学习的参数配置基于yaml配置文件，在特定角色的上需要设置不同配置，详细信息参见下表：

| 功能分类              | 配置参数                                     | 联邦学习角色 |
|-------------------|------------------------------------------| ------------ |
| global            | checkpoint_dir                           | server       |
|                   | fl_name                                  | server       |
|                   | fl_iteration_num                         | server       |
|                   | server_mode                              | server       |
|                   | enable_ssl                               | server       |
| distributed_cache | type                                     | server       |
|                   | address                                  | server       |
|                   | plugin_lib_path                          | server       |
|                   | cacert_filename                          | server       |
|                   | capath                                   | server       |
|                   | cert_filename                            | server       |
|                   | private_key_filename                     | server       |
|                   | server_name                              | server       |
| round             | start_fl_job_threshold                   | server       |
|                   | start_fl_job_time_window                 | server       |
|                   | update_model_ratio                       | server       |
|                   | update_model_time_window                 | server       |
|                   | global_iteration_time_window             | server       |
| summary           | metrics_file                             | server       |
|                   | failure_event_file                       | server       |
|                   | continuous_failure_times                 | server       |
|                   | data_rate_dir                            | server       |
|                   | participation_time_level                 | server       |
| unsupervised      | cluster_client_num                       | server       |
|                   | eval_type                                | server       |
| encrypt           | encrypt_type                             | server       |
|                   | pw_encrypt.share_secrets_ratio           | server       |
|                   | pw_encrypt.cipher_time_window            | server       |
|                   | pw_encrypt.reconstruct_secrets_threshold | server       |
|                   | dp_encrypt.dp_eps                        | server       |
|                   | dp_encrypt.dp_delta                      | server       |
|                   | dp_encrypt.dp_norm_clip                  | server       |
|                   | signds.sign_k                            | server       |
|                   | signds.sign_eps                          | server       |
|                   | signds.sign_thr_ratio                    | server       |
|                   | signds.sign_global_lr                    | server       |
|                   | signds.sign_dim_out                      | server       |
| compression       | upload_compress_type                     | server       |
|                   | upload_sparse_rate                       | server       |
|                   | download_compress_type                   | server       |
| ssl               | server_cert_path                         | server       |
|                   | client_cert_path                         | server       |
|                   | ca_cert_path                             | server       |
|                   | crl_path                                 | server       |
|                   | cipher_list                              | server       |
|                   | cert_expire_warning_time_in_day          | server       |
| client_verify     | pki_verify                               | server       |
|                   | root_first_ca_path                       | server       |
|                   | root_second_ca_path                      | server       |
|                   | equip_crl_path                           | server       |
|                   | replay_attack_time_diff                  | server       |
| client            | http_url_prefix                          | server       |
|                   | client_epoch_num                         | server       |
|                   | client_batch_size                        | server       |
|                   | client_learning_rate                     | server       |
|                   | connection_num                           | server       |

其中：

- **checkpoint_dir** (str) - server读取和保存模型文件的目录。默认值：``'./fl_ckpt/'``。
- **fl_name** (str) - 联邦学习作业名称。默认值：``"Lenet"``。
- **fl_iteration_num** (int) - 联邦学习的迭代次数，即客户端和服务器的交互次数。默认值：``20``。
- **server_mode** (str) - 描述服务器模式，它必须是 ``'FEDERATED_LEARNING'``，``'CLOUD_TRAINING'`` 和 ``'HYBRID_TRAINING'`` 中的一个。
- **enable_ssl** (bool) - 设置联邦学习开启SSL安全通信。默认值：``False``。
- **type**(str) - 使用的分布式缓存数据库，默认值：``"redis"``。
- **address**  - (str) - 设置分布式缓存数据库的地址，格式为ip:port，默认值：``'127.0.0.1：2345'``。
- **plugin_lib_path** (str) - 第三方插件路径，默认值： ``""``。
- **cacert_filename** (str) - 当ssl=true时配置，根证书文件路径， 默认值： ``""``。
- **capath** (str) - 根证书文件路径。默认值，默认值： ``""``。
- **cert_filename** (str) - 根证书文件路径，默认值： ``""``。
- **private_key_filename** (str) - 证书私钥文件路径，默认值： ``""``。
- **server_name** (str) - 服务器名称，默认值： ``""``。
- **start_fl_job_threshold** (int) - 开启联邦学习作业的阈值计数。默认值：``1``。
- **start_fl_job_time_window** (int) - 开启联邦学习作业的时间窗口持续时间，以毫秒为单位。默认值：``300000``。
- **update_model_ratio** (float) - 计算更新模型阈值计数的比率。默认值：``1.0``。
- **update_model_time_window** (int) - 更新模型的时间窗口持续时间，以毫秒为单位。默认值：``300000``。
- **metrics_file** (str) -  用于记录metrics集群运行训练指标信息，默认值：``"metrics.json"``。
- **failure_event_file** (str) - 用于记录集群异常事件文件路径，默认值：``"event.txt"``。
- **continuous_failure_times** (int) - 迭代失败次数大于该参数，统计失败事件，默认值：``10``。
- **data_rate_dir** (str) - 集群流量统计信息文件路径 ，默认值：``".."``。
- **participation_time_level** (str) - 流量统计时间区间，默认值：``"5,15"``。
- **encrypt_type** (str) - 用于联邦学习的安全策略，可以是 ``'NOT_ENCRYPT'``、``'DP_ENCRYPT'``、``'PW_ENCRYPT'``、``'STABLE_PW_ENCRYPT'`` 或 ``'SIGNDS'``。如果是 ``'DP_ENCRYPT'``，则将对客户端应用差分隐私模式，隐私保护效果将由上面所述的 `dp_eps`、`dp_delta`、`dp_norm_clip` 确定。如果 ``'PW_ENCRYPT'``，则将应用成对（pairwise，PW）安全聚合来保护客户端模型在跨设备场景中不被窃取。如果 ``'STABLE_PW_ENCRYPT'``，则将应用成对安全聚合来保护客户端模型在云云联邦场景中免受窃取。如果 ``'SIGNDS'``，则将在于客户端上使用SignDS策略。SignDS的介绍可以参照：[SignDS-FL: Local Differentially Private Federated Learning with Sign-based Dimension Selection](https://dl.acm.org/doi/abs/10.1145/3517820)。默认值：``'NOT_ENCRYPT'``。
- **share_secrets_ratio** (float) - PW：参与秘密分享的客户端比例。默认值：``1.0``。
- **cipher_time_window** (int) - PW：每个加密轮次的时间窗口持续时间，以毫秒为单位。默认值：``300000``。
- **reconstruct_secrets_threshold** (int) - PW：秘密重建的阈值。默认值：``2000``。
- **dp_eps** (float) - DP：差分隐私机制的epsilon预算。`dp_eps` 越小，隐私保护效果越好。默认值：``50.0``。
- **dp_delta** (float) - DP：差分隐私机制的delta预算，通常等于客户端数量的倒数。dp_delta越小，隐私保护效果越好。默认值：``0.01``。
- **dp_norm_clip** (float) - DP：差分隐私梯度裁剪的控制因子。建议其值为0.5~2。默认值：``1.0``。
- **sign_k** (float) - SignDS：Top-k比率，即Top-k维度的数量除以维度总数。建议取值范围在(0, 0.25]内。默认值：``0.01``。
- **sign_eps** (float) - SignDS：隐私预算。该值越小隐私保护力度越大，精度越低。建议取值范围在(0, 100]内。默认值：``100``。
- **sign_thr_ratio** (float) - SignDS：预期Top-k维度的阈值。建议取值范围在[0.5, 1]内。默认值：``0.6``。
- **sign_global_lr** (float) - SignDS：分配给选定维的常量值。适度增大该值会提高收敛速度，但有可能让模型梯度爆炸。取值必须大于0。默认值：``1``。
- **sign_dim_out** (int) - SignDS：输出维度的数量。建议取值范围在[0, 50]内。默认值：``0``。
- **upload_compress_type** (str) - 上传压缩方法。可以是 ``'NO_COMPRESS'`` 或 ``'DIFF_SPARSE_QUANT'``。如果是 ``'NO_COMPRESS'``，则不对上传的模型进行压缩。如果是 ``'DIFF_SPARSE_QUANT'``，则对上传的模型使用权重差+稀疏+量化压缩策略。默认值：``'NO_COMPRESS'``。
- **upload_sparse_rate** (float) - 上传压缩稀疏率。稀疏率越大，则压缩率越小。取值范围：(0, 1.0]。默认值：``0.4``。
- **download_compress_type** (str) - 下载压缩方法。可以是 ``'NO_COMPRESS'`` 或 ``'QUANT'``。如果是 ``'NO_COMPRESS'``，则不对下载的模型进行压缩。如果是 ``'QUANT'``，则对下载的模型使用量化压缩策略。默认值：``'NO_COMPRESS'``。
- **server_cert_path** (str) - 云侧服务器证书文件路径，默认值：``"server.p12"``。
- **client_cert_path** (str) - 云侧客户端证书文件路径，默认值：``"client.p12"``。
- **ca_cert_path** (str) - 云侧根证书文件路径，默认值：``"ca.crt"``。
- **crl_path** (str) - 云侧crl证书文件路径，默认值： ``""``。
- **cipher_list** (str) - 云侧ssl支持的加密套件，默认值：``"ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-PSK-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-CCM:ECDHE-ECDSA-AES256-CCM:ECDHE-ECDSA-CHACHA20-POLY1305"``。
- **cert_expire_warning_time_in_day** (int) - 云侧证书到期前提示时间，默认值：``90``。
- **pki_verify** (bool) - 如果为 ``True``，则将打开服务器和客户端之间的身份验证。还应从[华为CBG数字证书下载管理中心](https://pki.consumer.huawei.com/ca/)下载Root CA证书、Root CA G2证书和移动设备CRL证书。需要注意的是，只有当客户端是具有HUKS服务的Android环境时，`pki_verify` 可以为 ``True``。默认值：``False``。
- **root_first_ca_path** (str) - Root CA证书的文件路径。当 `pki_verify` 为 ``True`` 时，需要设置该值。默认值： ``""``。
- **root_second_ca_path** (str) - Root CA G2证书的文件路径。当 `pki_verify` 为 ``True`` 时，需要设置该值。默认值： ``""``。
- **equip_crl_path** (str) - 移动设备CRL证书的文件路径。当 `pki_verify` 为 ``True`` 时，需要设置该值。默认值： ``""``。
- **replay_attack_time_diff** (int) - 证书时间戳验证的最大可容忍错误（毫秒）。默认值：``600000``。
- **http_url_prefix** (str) - 设置联邦学习端云通信的http路径。默认值： ``""``。
- **client_epoch_num** (int) - 客户端训练epoch数量。默认值：``25``。
- **client_batch_size** (int) - 客户端训练数据batch数。默认值：``32``。
- **client_learning_rate** (float) - 客户端训练学习率。默认值：``0.001``。
- **connection_num** (int) - 云侧可建立的tcp链接最大值，默认值：``10000``。
- **cluster_client_num** (int) - 云侧进行无监督聚类指标评价的group id数目，默认值：``1000``。
- **eval_type** (str) - 云侧进行无监督聚类指标评价的算法类型，默认值：``"NOT_EVAL"``。

如下提供Lenet的yaml配置做为实例参考：

```yaml
fl_name: Lenet
fl_iteration_num: 25
server_mode: FEDERATED_LEARNING
enable_ssl: False

distributed_cache:
  type: redis
  address: 10.113.216.40:23456
  plugin_lib_path: ""

round:
  start_fl_job_threshold: 2
  start_fl_job_time_window: 30000
  update_model_ratio: 1.0
  update_model_time_window: 30000
  global_iteration_time_window: 60000

summary:
  metrics_file: "metrics.json"
  failure_event_file: "event.txt"
  continuous_failure_times: 10
  data_rate_dir: ".."
  participation_time_level: "5,15"

unsupervised:
  cluster_client_num: 1000
  eval_type: SILHOUETTE_SCORE

encrypt:
  encrypt_type: NOT_ENCRYPT
  pw_encrypt:
    share_secrets_ratio: 1.0
    cipher_time_window: 3000
    reconstruct_secrets_threshold: 1
  dp_encrypt:
    dp_eps: 50.0
    dp_delta: 0.01
    dp_norm_clip: 1.0
  signds:
    sign_k: 0.01
    sign_eps: 100
    sign_thr_ratio: 0.6
    sign_global_lr: 0.1
    sign_dim_out: 0

compression:
  upload_compress_type: NO_COMPRESS
  upload_sparse_rate: 0.4
  download_compress_type: NO_COMPRESS

ssl:
  # when ssl_config is set
  # for tcp/http server
  server_cert_path: "server.p12"
  # for tcp client
  client_cert_path: "client.p12"
  # common
  ca_cert_path: "ca.crt"
  crl_path: ""
  cipher_list: "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-PSK-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-CCM:ECDHE-ECDSA-AES256-CCM:ECDHE-ECDSA-CHACHA20-POLY1305"
  cert_expire_warning_time_in_day: 90

client_verify:
  pki_verify: false
  root_first_ca_path: ""
  root_second_ca_path: ""
  equip_crl_path: ""
  replay_attack_time_diff: 600000

client:
  http_url_prefix: ""
  client_epoch_num: 20
  client_batch_size: 32
  client_learning_rate: 0.01
  connection_num: 10000

```
