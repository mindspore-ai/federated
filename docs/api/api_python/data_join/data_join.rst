mindspore_federated.data_join
================================

.. py:class:: mindspore_federated.data_join.FLDataWorker(role, worker_config_path, data_schema_path, server_address, peer_server_address)

    数据接入进程。

    参数：
        - **role** (str) - 进程的角色类型。支持["leader", "follower"]。
        - **worker_config_path** (str) - 求交时所需要配置的超参文件存放的路径。
        - **schema_path** (str) - 导出时所需要配置的超参文件存放的路径。
        - **server_address** (str) - 本机IP和端口地址。
        - **peer_server_address** (str) - 对端IP和端口地址。

    .. py:method:: export()

        根据交集导出MindRecord。


.. py:function:: mindspore_federated.data_join.load_mindrecord(input_dir, seed=0, **kwargs)

    读取MindRecord文件。

    参数：
        - **input_dir** (str) - 输入的MindRecord相关文件的目录。
        - **seed** (int) - 随机种子。

    返回：
        - **dataset** (MindDataset) - 保序的数据集。

    .. note::
        该接口将 `kwargs` 透传给MindDataset。
        有关 `kwargs` 中更多超参数的详细信息，请参见MindDataset。