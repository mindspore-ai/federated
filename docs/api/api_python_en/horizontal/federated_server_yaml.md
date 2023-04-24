# Yaml Detail Configuration Items

The configuration of the parameters for the existing federation learning is based on the yaml configuration file, and different configurations need to be set on specific roles. For details, check the following table.

| Function classification |Configuration parameters   | Federal Learning Roles       |
|---------------| ------------------------- |--------|
| global        | checkpoint_dir            | server |
|               | fl_name                   | server |
|               | fl_iteration_num          | server |
|               | server_mode               | server |
|               | enable_ssl                | server |
| distributed_cache | type                      | server |
|               | address                   | server |
|               | plugin_lib_path           | server |
|               | cacert_filename           | server |
|               | capath                    | server |
|               | cert_filename             | server |
|               | private_key_filename      | server |
|               | server_name               | server |
| round         | start_fl_job_threshold    | server |
|               | start_fl_job_time_window  | server |
|               | update_model_ratio        | server |
|               | update_model_time_window  | server |
|               | global_iteration_time_window | server |
| summary       | metrics_file              | server |
|               | failure_event_file        | server |
|               | continuous_failure_times  | server |
|               | data_rate_dir             | server |
|               | participation_time_level  | server |
| encrypt       | encrypt_type              | server |
|               | pw_encrypt.share_secrets_ratio | server |
|               | pw_encrypt.cipher_time_window | server |
|               | pw_encrypt.reconstruct_secrets_threshold | server |
|               | dp_encrypt.dp_eps         | server |
|               | dp_encrypt.dp_delta       | server |
|               | dp_encrypt.dp_norm_clip   | server |
|               | signds.sign_k             | server |
|               | signds.sign_eps           | server |
|               | signds.sign_thr_ratio     | server |
|               | signds.sign_global_lr     | server |
|               | signds.sign_dim_out       | server |
| compression   | upload_compress_type      | server |
|               | upload_sparse_rate        | server |
|               | download_compress_type    | server |
| ssl           | server_cert_path          | server |
|               | client_cert_path          | server |
|               | ca_cert_path              | server |
|               | crl_path                  | server |
|               | cipher_list               | server |
|               | cert_expire_warning_time_in_day | server |
| client_verify | pki_verify                | server |
|               | root_first_ca_path        | server |
|               | root_second_ca_path       | server |
|               | equip_crl_path            | server |
|               | replay_attack_time_diff   | server |
| client        | http_url_prefix           | server |
|               | client_epoch_num          | server |
|               | client_batch_size         | server |
|               | client_learning_rate      | server |
|               | connection_num            | server |

which:

- **checkpoint_dir** (str) - The directory where the server reads and saves model files. Default: ``'. /fl_ckpt/'``.
- **fl_name** (str) - The name of the federal learning job. Default: ``'Lenet'``.
- **fl_iteration_num** (int) - The number of iterations of federated learning, i.e. the number of client-server interactions. Default: ``20``.
- **server_mode** (str) - Describes the server mode. it must be one of ``'FEDERATED_LEARNING'`` and ``'HYBRID_TRAINING'``.
- **enable_ssl** (bool) - Sets federated learning to enable SSL secure communication. Default: ``False``.
- **type** (str) - The distributed cache database to use, default: ``'redis'``.
- **address** - (str) - Sets the address of the distributed cache database in the format ip:port, Default: ``'127.0.0.1: 2345'``.
- **plugin_lib_path** (str) - The path to the third-party plugin. Default: ``""``.
- **cacert_filename** (str) - Configured when ssl=true, path to root certificate file, default: ``""``.
- **capath** (str) - Root certificate file path default, Default: ``""``.
- **cert_filename** (str) - The path to the root certificate file, Default: ``""``.
- **private_key_filename** (str) - Path of certificate private key file, Default: ``""``.
- **server_name** (str) - Server name, Default: ``""``.
- **start_fl_job_threshold** (int) - The threshold count for opening a federal learning job. Default: ``1``.
- **start_fl_job_time_window** (int) - The duration of the time window in milliseconds to start a federated learning job. Default: ``300000``.
- **update_model_ratio** (float) - The ratio to calculate the update model threshold count. Default: ``1.0``.
- **update_model_time_window** (int) - The duration of the time window for updating the model, in milliseconds. Default: ``300000``.
- **metrics_file** (str) - Information for recording training metrics for metrics cluster runs, Default: ``"metrics.json"``.
- **failure_event_file** (str) - Path to the cluster exception event file, Default: ``"event.txt"``.
- **continuous_failure_times** (int) - The number of failed iterations greater than this parameter to count failed events, Default: ``10``.
- **data_rate_dir** (str) - The path to the cluster traffic statistics file , Default: ``"...". ."``.
- **participation_time_level** (str) - Traffic statistics time interval, Default: ``"5,15"``.
- **encrypt_type** (str) - The security policy used for federation learning, can be ``'NOT_ENCRYPT'``, ``'DP_ENCRYPT'``, ``'PW_ENCRYPT'``, ``'STABLE_PW_ENCRYPT'``, or ``'SIGNDS'``. If ``'DP_ENCRYPT'``, the differential privacy mode will be applied to the client and the privacy protection effect will be determined by dp_eps, dp_delta, dp_norm_clip as described above. If ``'PW_ENCRYPT'``, pairwise (PW) security aggregation will be applied to protect the client model from being stolen in cross-device scenarios. If ``'STABLE_PW_ENCRYPT'``, pairwise security aggregation will be applied to protect the client model from theft in a cloud federation scenario. If ``'SIGNDS'``, the SignDS policy will be used on the client. -based Dimension Selection](<https://dl.acm.org/doi/abs/10.1145/3517820>). Default: ``'NOT_ENCRYPT'``.
- **share_secrets_ratio** (float) - PW: The percentage of clients participating in secret sharing. Default: ``1.0``.
- **cipher_time_window** (int) - PW: duration of the time window for each encryption round in milliseconds. Default: ``300000``.
- **reconstruct_secrets_threshold** (int) - PW: The threshold for secret reconstruction. Default: ``2000``.
- **dp_eps** (float) - DP: epsilon budget of the differential privacy mechanism. The smaller the dp_eps, the better the privacy protection. Default value: ``50.0``.
- **dp_delta** (float) - DP: The delta budget of the differential privacy mechanism, usually equal to the inverse of the number of clients. the smaller the dp_delta, the better the privacy protection. Default: ``0.01``.
- **dp_norm_clip** (float) - DP: control factor for differential privacy gradient clipping. The recommended value is 0.5~2. Default: ``1.0``.
- **sign_k** (float) - SignDS: Top-k ratio, i.e., the number of Top-k dimensions divided by the total number of dimensions. Recommended values are in the range of (0, 0.25]. Default: ``0.01``.
- **sign_eps** (float) - SignDS: privacy budget. The smaller the value, the stronger the privacy protection, and the lower the precision. Recommended values are in the range (0, 100]. Default value: ``100``.
- **sign_thr_ratio** (float) - SignDS: The threshold value for the expected Top-k dimension. Recommended values are in the range [0.5, 1]. Default: ``0.6``.
- **sign_global_lr** (float) - SignDS: Constant value assigned to the selected dimension. Moderately increasing this value will improve the convergence speed, but may explode the model gradient. The value must be greater than 0. Default: ``1``.
- **sign_dim_out** (int) - SignDS: the number of output dimensions. Recommended values are in the range [0, 50]. Default: ``0``.
- **upload_compress_type** (str) - Upload compression method. Can be ``'NO_COMPRESS'`` or ``'DIFF_SPARSE_QUANT'``. If it is ``'NO_COMPRESS'``, no compression is applied to the uploaded model. If it is ``'DIFF_SPARSE_QUANT'``, the uploaded model is compressed using the weight difference + sparse + quantized compression strategy. Default value: ``'NO_COMPRESS'``.
- **upload_sparse_rate** (float) - The upload compression sparsity rate. The larger the sparse rate, the smaller the compression rate. Value range: (0, 1.0]. Default: ``0.4``.
- **download_compress_type** (str) - The download compression method. Can be 'NO_COMPRESS' or 'QUANT'. If it is 'NO_COMPRESS', the downloaded model will not be compressed. If it is 'QUANT', the quantitative compression strategy is used for the downloaded models. Default: ``'NO_COMPRESS'``.
- **server_cert_path** (str) - The path to the cloud-side server certificate file, Default: ``'server.p12'``.
- **client_cert_path** (str) - Cloud-side client certificate file path, Default: ``"client.p12"``.
- **ca_cert_path** (str) - The path to the cloud-side root certificate file, Default: ``"ca.crt"``.
- **crl_path** (str) - The path to the cloud-side crl certificate file, Default: ``""``.
- **cipher_list** (str) - The cipher suite supported by cloud-side ssl, Default: ``"ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384: ecdhe-ecdsa-aes256-gcm-sha384:ecdhe-rsa-chacha20-poly1305:ecdhe-psk-chacha20-poly1305:ecdhe-ecdsa-aes128-ccm:ecdhe-ecdsa-aes256- CCM:ECDHE-ECDSA-CHACHA20-POLY1305"``.
- **cert_expire_warning_time_in_day** (int) - Time to prompt before the cloud-side certificate expires, Default: ``90``.
- **pki_verify** (bool) - If ``True``, authentication between server and client will be turned on. Root CA certificate, Root CA G2 certificate and mobile device CRL certificate should also be downloaded from [Huawei CBG Digital Certificate Download Management Center](https://pki.consumer.huawei.com/ca/). Note that `pki_verify` can be ``True`` only if the client is an Android environment with HUKS service. Default: ``False``.
- **root_first_ca_path** (str) - The file path of the Root CA certificate. This value needs to be set when `pki_verify` is ``True``. Default: ``""``.
- **root_second_ca_path** (str) - The file path of the Root CA G2 certificate. This value needs to be set when `pki_verify` is ``True``. Default: ``""``.
- **equip_crl_path** (str) - The file path of the mobile device CRL certificate. This value needs to be set when `pki_verify` is ``True``. Default: ``""``.
- **replay_attack_time_diff** (int) - The maximum tolerable error (in milliseconds) for certificate timestamp verification. Default: ``600000``.
- **http_url_prefix** (str) - Sets the http path to the federal learning end cloud communication. Default: ``""``.
- **client_epoch_num** (int) - The number of client training epochs. Default: ``25``.
- **client_batch_size** (int) - The number of client training data batches. Default: ``32``.
- **client_learning_rate** (float) - The client training learning rate. Default: ``0.001``.
- **connection_num** (int) - Maximum number of tcp links that can be established on the cloud side, Default: ``10000``.

The following yaml configuration of Lenet is provided as an example reference.

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
