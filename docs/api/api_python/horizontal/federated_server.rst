服务器启动接口
======================

.. py:class:: mindspore_federated.FLServerJob(yaml_config, http_server_address, tcp_server_ip="127.0.0.1", checkpoint_dir="./fl_ckpt/", ssl_config=None)

    定义联邦学习云侧任务。

    参数：
        - **yaml_config** (str) - yaml文件路径。更多细节见 `yaml配置说明 <https://gitee.com/mindspore/federated/blob/master/docs/api/api_python/horizontal/federated_server_yaml.md>`_。
        - **http_server_address** (str) - 用于通信的http服务器地址。
        - **tcp_server_ip** (str) - 用于通信的tcp服务器地址。默认值： ``'127.0.0.1'``。
        - **checkpoint_dir** (str) - 存储权重的路径。默认值： ``"./fl_ckpt/"``。
        - **ssl_config** (Union(None, SSLConfig)) - ssl配置项。默认值：``None``。

    .. py:method:: run(feature_map=None, callback=None)

        运行联邦学习服务器任务。

        参数：
            - **feature_map** (Union(dict, FeatureMap, str)) - 特征图。默认值：``None``。
            - **callback** (Union(None, Callback)) - 回调函数。默认值：``None``。

.. py:class:: mindspore_federated.FlSchedulerJob(yaml_config, manage_address, ssl_config=None)

    定义联邦学习调度任务。

    参数：
        - **yaml_config** (str) - yaml文件路径。更多细节见 `yaml配置说明 <https://gitee.com/mindspore/federated/blob/master/docs/api/api_python/horizontal/federated_server_yaml.md>`_。
        - **manage_address** (str) - 管理地址。
        - **ssl_config** (Union(None, SSLConfig)) - ssl配置项。默认值：``None``。

    .. py:method:: run()

        运行调度任务。

