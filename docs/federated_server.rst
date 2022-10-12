Federated-Server
======================

.. py:class:: mindspore.FLServerJob(yaml_config, http_server_address, tcp_server_ip="127.0.0.1", checkpoint_dir="./fl_ckpt/", ssl_config=None)

    定义联邦学习云侧任务。

    参数：
        - **yaml_config** (str) - yaml文件路径。更多细节见 https://gitee.com/mindspore/federated/blob/master/docs/federated_server_yaml.md。
        - **http_server_address** (str) - 用于通信的http服务器地址。
        - **tcp_server_ip** (str) - 用于通信的tcp服务器地址。默认为值：127.0.0.1。
        - **checkpoint_dir** (str) - 存储权重的路径。默认为值：./fl_ckpt/。
        - **ssl_config** (Union(None, SSLConfig)) - ssl配置项。默认值：None。

    .. py:method:: run(feature_map=None, callback=None)

        运行云侧任务。

        参数：
            - **feature_map** (Union(dict, FeatureMap, str)) - 特征集。
            - **callback** (Union(None, Callback)) - 回调函数。

    .. py:method:: after_started_callback()

        定义联邦任务开始后的回调函数。

    .. py:method:: before_stopped_callback()

        定义联邦任务结束后的回调函数。

    .. py:method:: on_iteration_end_callback(feature_list, fl_name, instance_name, iteration_num,
                                  iteration_valid, iteration_reason)

        定义迭代结束后的回调函数。

        参数：
            - **feature_list** (list) - 特征集。
            - **fl_name** (str) - 当前联邦学习的名称。
            - **instance_name** (str) - 当前实例名称。
            - **iteration_valid** (int) - 开启验证的迭代数。
            - **iteration_reason** (str) - 开启验证原因。

.. py:class:: mindspore.FlSchedulerJob(yaml_config, manage_address, ssl_config=None)

    定义联邦学习调度任务。

    参数：
        - **yaml_config** (str) - yaml文件路径。更多细节见 https://gitee.com/mindspore/federated/blob/master/docs/federated_server_yaml.md。
        - **manage_address** (str) - 管理地址。
        - **ssl_config** (Union(None, SSLConfig)) - ssl配置项。默认值：None。

    .. py:method:: run(feature_map=None, callback=None)

        运行调度任务。

