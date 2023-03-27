模型训练接口
==========================================

.. py:class:: mindspore_federated.FLModel(yaml_data, network, loss_fn=None, optimizers=None, metrics=None, eval_network=None)

    用于纵向联邦学习模型训练与推理的高阶API。FLModel将纵向联邦学习参与方训练所需的网络模型、优化器，以及其它数据结构封装为高阶对象。然后，FLModel根据开发者配置的yaml文件（参见 `纵向联邦学习yaml详细配置项 <https://gitee.com/mindspore/federated/blob/master/docs/api/api_python/vertical/vertical_federated_yaml.md>`_ ），构建纵向联邦学习流程，并提供控制训练和推理流程的接口。

    参数：
        - **yaml_data** (class) - 包含纵向联邦学习流程相关信息的数据类，包括优化器、梯度计算器等模块信息。该数据类从开发者配置的yaml文件解析得到。
        - **network** (Cell) - 训练网络，输出loss值。如果未输入 `loss_fn` ，则直接采用该网络作为训练网络。如果输入了有效的 `loss_fn` ，则将基于 `network` 和 `loss_fn` 构建训练网络。
        - **loss_fn** (Cell) - 损失函数。若初始化阶段未指定loss_fn，则输入的network将直接被用作训练网络。默认值：None。
        - **optimizers** (Cell) - 用于训练训练网络的自定义优化器。若 `optimizers` 为None，FLModel将根据yaml文件配置信息，使用MindSpore提供的标准优化器训练训练网络。默认值：None。
        - **metrics** (Metric) - 用于计算评估网络评测指标的类。默认值：None。
        - **eval_network** (Cell) - 参与方评估网络，输出任务预测值。默认值：None。

    .. py:method:: backward_one_step(local_data_batch: dict = None, remote_data_batch: dict = None, sens: dict = None)

        采用一个数据batch，执行训练网络反向传播。

        参数：
            - **local_data_batch** (dict) - 从本地服务器读取的数据batch。key为数据项名称，value为对应的Tensor。默认值：None。
            - **remote_data_batch** (dict) - 从其它参与方的远程服务器读取的数据batch。key为数据项名称，value为对应的Tensor。默认值：None。
            - **sens** (dict) - 用于训练网络梯度值计算的sens加权系数。其key为yaml文件中定义的sens加权系数名称，其value为包含sens加权系数Tensor的字典。value字典的key为训练网络的输出Tensor名称，value字典的value为该输出对应的sens加权系数Tensor。默认值：None。

        返回：
            Dict，传递给其它纵向联邦学习参与方，用于其梯度值计算的sens加权系数。其key为yaml文件中定义的sens加权系数名称，其value为包含sens加权系数Tensor的字典。value字典的key为训练网络的输入Tensor名称，value字典的value为该输入对应的sens加权系数Tensor。


    .. py:method:: eval_one_step(local_data_batch: dict = None, remote_data_batch: dict = None)

        采用一个数据batch，执行评估网络计算。

        参数：
            - **local_data_batch** (dict) - 从本地服务器读取的数据batch。key为数据项名称，value为对应的Tensor。默认值：None。
            - **remote_data_batch** (dict) - 从其它参与方的远程服务器读取的数据batch。key为数据项名称，value为对应的Tensor。默认值：None。

        返回：
            Dict，评估网络的输出。key为评估网络输出的变量名称，value为对应的Tensor。


    .. py:method:: forward_one_step(local_data_batch: dict = None, remote_data_batch: dict = None)

        采用一个数据batch，执行训练网络前向推理。

        参数：
            - **local_data_batch** (dict) - 从本地服务器读取的数据batch。key为数据项名称，value为对应的Tensor。默认值：None。
            - **remote_data_batch** (dict) - 从其它参与方的远程服务器读取的数据batch。key为数据项名称，value为对应的Tensor。默认值：None。

        返回：
            Dict，评估网络的输出。key为训练网络输出的变量名称，value为对应的Tensor。

    .. py:method:: get_compress_configs()

        获取压缩配置。

        .. note::
            无法给名字相同的tensors使用不同的压缩方法。

        返回：
            Dict, Key是tensor的名字, Value是tensor。

    .. py:method:: load_ckpt(phrase: str = 'eval', path: str = None)

        加载checkpoint至训练网络和评估网络。

        参数：
            - **phrase** (str) - 加载checkpoint至哪个网络，必须为'eval'或'train'。如果设置为'eval'，加载checkpoint至评估网络；如果设置为'train'，加载checkpoint至训练网络。默认值：'eval'。
            - **path** (str) - 加载checkpoint的路径。如果未定义 `path` ，则将使用yaml文件中定义的 `ckpt_path` 作为checkpoint保存路径。默认值：None。


    .. py:method:: save_ckpt(path: str = None)

        保存训练网络的checkpoint。

        参数：
            - **path** (str) - 保存checkpoint的路径。如果未定义 `path` ，则将使用yaml文件中定义的 `ckpt_path` 作为checkpoint保存路径。默认值：None。

.. py:class:: mindspore_federated.FLYamlData(yaml_path: str)

    储存纵向联邦学习流程相关的配置信息，包括网络、优化器、算子等模块的输入/输出和超参数。上述信息从开发者提供的yaml文件（参见 `纵向联邦学习yaml详细配置项 <https://gitee.com/mindspore/federated/blob/master/docs/api/api_python/vertical/vertical_federated_yaml.md>`_ ）中解析上述配置信息。在解析过程中，将会校验yaml文件的合法性。返回值给FLModel第一个入参使用。

    参数：
        - **yaml_path** (str) - yaml文件路径。
