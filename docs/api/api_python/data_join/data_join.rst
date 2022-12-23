data_join
================================

.. py:class:: mindspore_federated.data_join.FLDataWorker(role, main_table_files, output_dir, data_schema_path,communicator, primary_key="oaid", bucket_num=5, store_type="csv", shard_num=1, join_type="psi", thread_num=0)

    数据接入进程。

    参数：
        - **role** (str) - 进程的角色类型。支持["leader", "follower"]。
        - **main_table_files** (Union(list(str), str) - 原始文件路径，必须在leader和follower上都设置。
        - **output_dir** (str) - 输出目录，必须在leader和follower上都设置。
        - **data_schema_path** (str) - 数据的schema的路径，必须在leader和follower上都设置。
          用户需要在schema中提供要导出的数据的列名和类型。schema需要被解析为key-value形式的双层字典。
          第一级字典的key值为列名，value值为第二级字典。
          第二级字典的key值为字符串类型的”type“，value值为字段对应数据被导出时所保存的类型。
          当前支持的类型包括：["int32", "int64", "float32", "float64", "string", "bytes"]。
        - **communicator** (VerticalFederatedCommunicator) - 纵向联邦框架的Http与Https通信器。
        - **primary_key** (str) - 主键名称。leader侧设置的值被使用，follower设置的值无效。默认值：oaid。
        - **bucket_num** (int) - 桶的数目。leader侧设置的值被使用，follower设置的值无效。默认值：1。
        - **store_type** (str) - 数据的存储类型。默认值：csv。
        - **shard_num** (int) - 每个桶导出的文件个数。leader侧设置的值被使用，follower设置的值无效。默认值：1。
        - **join_type** (str) - 求交类型。leader侧设置的值被使用，follower设置的值无效。默认值：psi。
        - **thread_num** (int) - psi中的线程数目。支持["leader", "follower"]。默认值：0。

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