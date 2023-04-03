psi
================================

.. py:function:: RunPSI(input_data, comm_role, peer_comm_role, bucket_id, thread_num)

    密文求交函数。

    .. note::
        需要先通过 `from mindspore_federated._mindspore_federated import RunPSI` 导入该函数；
        在调用本接口前，需要初始化纵向联邦通信实例，可参考 `MindSpore federated ST <https://gitee.com/mindspore/federated/blob/r0.1/tests/st/psi/run_psi.py>`_ 。

    参数：
        - **input_data** (list[string]) - 己方的输入数据。
        - **comm_role** (string) - 该进程的通信角色，"server" 或 "client"。
        - **peer_comm_role** (string) - 对方的通信角色，"server" 或 "client"。
        - **bucket_id** (int) - 桶序号。双进程通信时，若双方该值不同，server 报错退出，client 阻塞等待。
        - **thread_num** (int) - 线程数目。0 表示使用机器最大可用线程数目减 5，其他值会限定在 1 到机器最大可使用值。

    返回：
        - **result** (list[string]) - 交集结果。

    异常：
        - **TypeError** - 输入 `input_data` 不是 list[string] 类型。
        - **TypeError** - 输入 `bucket_id` 不是大于等于0的整数类型，如负数或小数。
        - **TypeError** - 输入 `thread_num` 不是大于等于0的整数类型，如负数或小数。

    样例：
        >>> from mindspore_federated import VerticalFederatedCommunicator, ServerConfig
        >>> from mindspore_federated._mindspore_federated import RunPSI
        >>> http_server_config = ServerConfig(server_name='server', server_address="127.0.0.1:1086")
        >>> remote_server_config = ServerConfig(server_name='client', server_address="127.0.0.1:1087")
        >>> vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
        ...                                                       remote_server_config=remote_server_config)
        >>> vertical_communicator.launch()
        >>> result = RunPSI(['1', '2', '3'], 'server', 'client', 0, 0)


.. py:function:: PlainIntersection(input_data, comm_role, peer_comm_role, bucket_id, thread_num)

    明文求交函数。

    .. note::
        需要先通过 `from mindspore_federated._mindspore_federated import PlainIntersection` 导入该函数；
        在调用本接口前，需要初始化纵向联邦通信实例，可参考 `MindSpore federated ST <https://gitee.com/mindspore/federated/blob/r0.1/tests/st/psi/run_psi.py>`_ 。

    参数：
        - **input_data** (list[string]) - 己方的输入数据。
        - **comm_role** (string) - 该进程的通信角色，"server" 或 "client"。
        - **peer_comm_role** (string) - 对方的通信角色，"server" 或 "client"。
        - **bucket_id** (int) - 桶序号。双进程通信时，若双方该值不同，server 报错退出，client 阻塞等待。
        - **thread_num** (int) - 线程数目。0 表示使用机器最大可用线程数目减 5，其他值会限定在 1 到机器最大可使用值。

    返回：
        - **result** (list[string]) - 交集结果。

    异常：
        - **TypeError** - 输入 `input_data` 不是 list[string] 类型。
        - **TypeError** - 输入 `bucket_id` 不是大于等于0的整数类型，如负数或小数。
        - **TypeError** - 输入 `thread_num` 不是大于等于0的整数类型，如负数或小数。

    样例：
        >>> from mindspore_federated import VerticalFederatedCommunicator, ServerConfig
        >>> from mindspore_federated._mindspore_federated import PlainIntersection
        >>> http_server_config = ServerConfig(server_name='server', server_address="127.0.0.1:1086")
        >>> remote_server_config = ServerConfig(server_name='client', server_address="127.0.0.1:1087")
        >>> vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
        ...                                                       remote_server_config=remote_server_config)
        >>> vertical_communicator.launch()
        >>> result = PlainIntersection(['1', '2', '3'], 'server', 'client', 0, 0)