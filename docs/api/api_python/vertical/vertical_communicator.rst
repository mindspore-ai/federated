纵向联邦学习通信器
======================

.. py:class:: mindspore_federated.VerticalFederatedCommunicator(http_server_config, remote_server_config, ssl_config)

    定义纵向联邦学习通信器。

    参数：
        - **http_server_config** (ServerConfig) - 本地服务器配置。
        - **remote_server_config** (ServerConfig) - 远程服务器配置。
        - **ssl_config** (SSLConfig) - SSL加密通信配置。

    .. py:method:: data_join_wait_for_start()

        阻塞等待client worker的注册信息。

    .. py:method:: http_server_config()

        返回本地服务器配置。

    .. py:method:: launch()

        启动纵向联邦学习通信器。

    .. py:method:: receive(target_server_name)

        获取远程发送的Tensor数据。

        参数：
            - **target_server_name** (str) - 指定远程服务器名字。

    .. py:method:: remote_server_config()

       返回远端服务器配置。

    .. py:method:: send_register(target_server_name, worker_register)

        发送worker注册消息。

        参数：
            - **target_server_name** (str) - 指定远程服务器名字。
            - **worker_register** (_WorkerRegister) - 需要发送的worker注册信息。

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

    .. py:method:: init_server_config(http_server_config, remote_server_config)

        初始化本地服务器配置与远端服务器配置。

        参数：
            - **http_server_config** (ServerConfig) - 本地服务器配置。
            - **remote_server_config** (ServerConfig) - 远程服务器配置。
