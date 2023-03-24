纵向联邦学习通信器
======================

.. py:class:: mindspore_federated.VerticalFederatedCommunicator(http_server_config, remote_server_config, enable_ssl=False, ssl_config=None, compress_configs=None)

    定义纵向联邦学习通信器。

    参数：
        - **http_server_config** (ServerConfig) - 本地服务器配置。
        - **remote_server_config** (ServerConfig) - 远程服务器配置。
        - **enable_ssl** (bool) - 是否开启SSL加密通信。默认值：False。
        - **ssl_config** (SSLConfig) - SSL加密通信配置，默认值：None。
        - **compress_configs** (dict) - 通信压缩配置。默认值：None。

    .. py:method:: data_join_wait_for_start()

        阻塞等待client worker的注册信息。

    .. py:method:: http_server_config()

        返回本地服务器配置。

    .. py:method:: launch()

        启动纵向联邦学习通信器。

    .. py:method:: receive(target_server_name: str)

        获取远程发送的Tensor数据。

        参数：
            - **target_server_name** (str) - 指定远程服务器名字。

    .. py:method:: remote_server_config()

       返回远端服务器配置。

    .. py:method:: send_tensors(target_server_name, tensor_dict)

        发送分布式训练Tensor数据。

        参数：
            - **target_server_name** (str) - 指定远程服务器名字。
            - **tensor_dict** (OrderedDict) - 需要发送的Tensor字典。

.. py:class:: mindspore_federated.ServerConfig(server_name, server_address)

    定义纵向联邦服务器配置。

    参数：
        - **server_name** (str) - 服务器名字，比如leader_server，用户可自定义。
        - **server_address** (str) - 服务器地址，比如'127.0.0.1:1086'，用户可自定义。

.. py:class:: mindspore_federated.SSLConfig(server_password, client_password, server_cert_path=None, client_cert_path=None, ca_cert_path=None, crl_path=None, cipher_list=default_cipher_list, cert_expire_warning_time_in_day=90)

    定义纵向联邦服务器SSL通信配置。若用户想开启SSL需要配置如下参数。返回值给 `mindspore_federated.VerticalFederatedCommunicator` 的第三个入参使用。

    参数：
        - **server_password** (str) - 服务器证书的密码。默认为None。
        - **client_password** (str) - 客户端证书的密码。默认为None。
        - **server_cert_path** (str) - 服务器证书在服务器上的绝对路径。默认为None。
        - **client_cert_path** (str) - 客户端证书在服务器上的绝对路径。默认为None。
        - **ca_cert_path** (str) - 根证书在服务器上的绝对路径。默认为None。
        - **crl_path** (str) - CRL证书在服务器上的绝对路径。默认为None。
        - **cipher_list** (str) - 服务器支持ssl通信的默认加密套件。默认为"ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-PSK-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-CCM:ECDHE-ECDSA-AES256-CCM:ECDHE-ECDSA-CHACHA20-POLY1305"。
        - **cert_expire_warning_time_in_day** (int) - 证书在过期前多少天开始打印警告信息。默认为90。

.. py:class:: mindspore_federated.CompressConfig(compress_type, bit_num=8)

    定义纵向联邦服务器通信压缩配置。

    参数：
        - **compress_type** (str) - 纵向联邦通信压缩类型。支持["min_max", "bit_pack"]。

          - min_max：最小最大量化压缩方法。
          - bit_pack：比特打包压缩方法。

        - **bit_num** (int) - 量化算法的比特数，取值范围在[1, 8]内。默认值：8。