data_join
================================

.. py:class:: mindspore_federated.FLDataWorker(config: dict)

    和横向联邦学习不同，纵向联邦学习训练或推理时，两个参与方（leader和follower）拥有相同样本空间。因此，在纵向联邦学习的双方发起训练或推理之前，必须协同完成数据求交。双方必须读取各自的原始数据，并提取出每条数据对应的ID（每条数据的唯一标识符，且都不相同）进行求交（即求取交集）。然后，双方根据求交后的ID从原始数据中获得特征或标签等数据。最后各自导出持久化文件，并在后续训练或推理之前保序地读取数据。数据接入进程被用来导出数据。

    参数：
        - **config** (dict) - 输入参数字典，详细说明见下面各参数介绍。

          - role (str): 进程的角色类型。支持["leader", "follower"]。
          - bucket_num (int): 桶的数目。leader侧设置的值被使用，follower设置的值无效。
          - store_type (str): 数据的存储类型。当前仅支持csv和mysql。
          - data_schema_path (str): 数据的schema的路径，必须在leader和follower上都设置。
            用户需要在schema中提供要导出的数据的列名和类型。schema需要被解析为key-value形式的双层字典。
            第一级字典的key值为列名，value值为第二级字典。
            第二级字典的key值为字符串类型的”type“，value值为字段对应数据被导出时所保存的类型。
            当前支持的类型包括：["int32", "int64", "float32", "float64", "string", "bytes"]。
          - primary_key (str): 主键名称。leader侧设置的值被使用，follower设置的值无效。
          - main_table_files (Union(list(str), str)): 原始文件路径，必须在leader和follower上都设置。
          - mysql_host (str): MySql 服务器地址。
          - mysql_port (int): MySql 服务端口, 通常使用3306。
          - mysql_database (str): MySql数据库名, None 表示不指定。
          - mysql_charset (str): 使用的字符集。
          - mysql_user (str): 登录数据库的用户名。
          - mysql_password (str): 登录数据库的密码。
          - mysql_table_name (str): 存储原始数据的数据库表。
          - server_name (str): 通信时本地使用的http(s)服务名字。
          - http_server_address (str): 通信时本地使用的IP和端口，主/从节点均需要配置。
          - remote_server_name (str): 通信时对端使用的http(s)服务名字。
          - remote_server_address (str): 通信时对端使用的IP和端口，主/从节点均需要配置。
          - enable_ssl (bool): 通信模块是否开启SSL。 取值[True, False]。
          - server_password (str): 服务端P12证书文件保护口令，安全考虑请在命令行使用该密码。
          - client_password (str): 客户端P12证书文件保护口令，安全考虑请在命令行使用该密码。
          - server_cert_path (str): 服务端证书文件路径。
          - client_cert_path (str): 客户端证书文件路径。
          - ca_cert_path (str): CA证书文件路径。
          - crl_path (str): CRL 证书文件路径。
          - cipher_list (str): SSL加密算法清单。
          - cert_expire_warning_time_in_day (str): 证书过期预警时间。
          - join_type (str): 求交类型。leader侧设置的值被使用，follower设置的值无效。当前只支持psi。
          - thread_num (int): psi中的线程数目。
          - output_dir (str): 输出目录，必须在leader和follower上都设置。
          - shard_num (int): 每个桶导出的文件个数。leader侧设置的值被使用，follower设置的值无效。

          更多细节请参考 `vfl_data_join_config <https://e.gitee.com/mind_spore/repos/mindspore/federated/tree/master/tests/st/data_join/vfl/vfl_data_join_config.yaml>`_。


    .. py:method:: communicator()

        如果用户希望在数据接入和纵向联邦模型训练中使用同一个通信器，可以使用这个函数获得通信器实例。


    .. py:method:: do_worker()

        根据配置执行数据求交工作。


.. py:function:: mindspore_federated.data_join.load_mindrecord(input_dir, seed=0, **kwargs)

    读取MindRecord文件。

    参数：
        - **input_dir** (str) - 输入的MindRecord相关文件的目录。
        - **seed** (int) - 随机种子。默认值：0。

    返回：
        - **dataset** (MindDataset) - 保序的数据集。

    .. note::
        该接口将 `kwargs` 透传给MindDataset。有关 `kwargs` 中更多超参数的详细信息，参见 `mindspore.dataset.MindDataset` 。

.. py:function:: mindspore_federated.common.config.get_config(cfg_file)

    解析yaml文件获取配置信息。

    参数：
        - **cfg_file** (str) - yaml配置文件的路径。

    返回：
        argparse，解析yaml文件得到的配置信息。

    .. note::
        通过该接口获取字典格式的参数列表构造FLDataWorker。