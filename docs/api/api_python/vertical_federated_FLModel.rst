纵向联邦学习模型训练接口
==========================================

.. py:class:: mindspore_federated.FLModel(yaml_data, network, train_network=None, loss_fn=None, optimizers=None, metrics=None, eval_network=None, eval_indexes=None, grad_network=None)

    用于纵向联邦学习模型训练与推理的高阶API。FLModel将纵向联邦学习参与方训练所需的网络模型、优化器，以及其它数据结构封装为高阶对象.然后，FLModel根据开发者配置的yaml文件（参见[纵向联邦学习yaml详细配置项](https://gitee.com/mindspore/federated/blob/master/docs/api/api_python/vertical_federated_yaml.rst) ），构建纵向联邦学习流程，并提供控制训练和推理流程的接口。

    参数：
        - **yaml_data** (class) - 包含纵向联邦学习流程相关信息的数据类，包括优化器、梯度计算器等模块信息。该数据类从开发者配置的yaml文件解析得到。
        - **network** (Cell) - 参与方用于纵向联邦学习的基础网络，其参数被训练网络和评估网络共享。
        - **train_network** (Cell) - 参与方训练网络，输出损失值。若初始化阶段未指定train_network，则FLModel将使用network和loss_fn构造训练网络。默认值：None。
        - **loss_fn** (Cell) - 用于构建参与方训练网络的损失函数。如果train_network已被指定，则即使指定了loss_fn也不会被使用。默认值：None。
        - **optimizer** (Cell) - 用于训练训练网络的自定义优化器。若初始化阶段未指定optimizer，FLModel将根据yaml文件配置信息，使用MindSpore提供的标准优化器训练训练网络。默认值：None。
        - **metrics** (Metric) - 用于计算评估网络评测指标的类。默认值：None。
        - **eval_network** (Cell) - 参与方评估网络，输出任务预测值。默认值：None。
        - **grad_network** (Cell) - 运行于TEE可信执行环境，用于保护用户数据隐私的网络。默认值：None。

    .. py:method:: eval_one_step(local_data_batch=None, remote_data_batch=None)

        采用一个数据batch，执行评估网络前向推理。

        参数：
            - **local_data_batch** (dict) - 从本地服务器读取的数据batch。key为数据项名称，value为对应的Tensor张量。
            - **remote_data_batch** (dict) - 从其它参与方的远程服务器读取的数据batch。key为数据项名称，value为对应的Tensor张量。

        返回：
            Dict，评估网络的输出。key为评估网络输出的变量名称，value为对应的Tensor张量。


    .. py:method:: forward_one_step(local_data_batch=None, remote_data_batch=None)

        采用一个数据batch，执行训练网络前向推理。

        参数：
            - **local_data_batch** (dict) - 从本地服务器读取的数据batch。key为数据项名称，value为对应的Tensor张量。
            - **remote_data_batch** (dict) - 从其它参与方的远程服务器读取的数据batch。key为数据项名称，value为对应的Tensor张量。

        返回：
            Dict，评估网络的输出。key为训练网络输出的变量名称，value为对应的Tensor张量。


    .. py:method:: backward_one_step(local_data_batch=None, remote_data_batch=None sens=None)

        采用一个数据batch，执行训练网络反向传播。

        参数：
            - **local_data_batch** (dict) - 从本地服务器读取的数据batch。key为数据项名称，value为对应的Tensor张量。
            - **remote_data_batch** (dict) - 从其它参与方的远程服务器读取的数据batch。key为数据项名称，value为对应的Tensor张量。
            - **sens** (dict) - 用于训练网络梯度值计算的sens加权系数。其key为yaml文件中定义的sens加权系数名称，其value为包含sens加权系数张量的字典。value字典的key为训练网络的输出张量名称，value字典的value为该输出对应的sens加权系数张量.

        返回：
            Dict，传递给其它纵向联邦学习参与方，用于其梯度值计算的sens加权系数。其key为yaml文件中定义的sens加权系数名称，其value为包含sens加权系数张量的字典。value字典的key为训练网络的输入张量名称，value字典的value为该输入对应的sens加权系数张量.


    .. py:method:: save_ckpt(path=None)

        保存训练网络的checkpoint。

        参数：
            - **path** (str) - 保存checkpoint的路径。如果未定义path，则将使用yaml文件中定义的ckpt_path作为checkpoint保存路径。Default：None。


    .. py:method:: load_ckpt(phrase='eval', path=None)

        加载checkpoint至训练网络和评估网络。

        参数：
            - **phrase** (str) - 加载checkpoint至哪个网络，必须为'eval'或'train'。如果设置为'eval'，加载checkpoint至评估网络；如果设置为'train'，加载checkpoint至训练网络。Default：'eval'。
            - **path** (str) - 加载checkpoint的路径。如果未定义path，则将使用yaml文件中定义的ckpt_path作为checkpoint保存路径。Default：None。

.. py:class:: mindspore_federated.FLYamlData(path)

    储存纵向联邦学习流程相关的配置信息，包括网络、优化器、算子等模块的输入/输出和超参数。上述信息从开发者提供的yaml文件（参见[纵向联邦学习yaml详细配置项](https://gitee.com/mindspore/federated/blob/master/docs/api/api_python/vertical_federated_yaml.rst) ）中解析上述配置信息。在解析过程中，将会校验yaml文件的合法性。

    参数：
        - **path** (str) - yaml文件路径。
