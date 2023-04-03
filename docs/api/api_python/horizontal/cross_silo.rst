云侧客户端
================================

.. py:class:: mindspore_federated.FederatedLearningManager(yaml_config, model, sync_frequency, http_server_address="", data_size=1, sync_type='fixed', run_distribute=False, ssl_config=None, **kwargs)

    在训练过程中管理联邦学习。

    参数：
        - **yaml_config** (str) - yaml文件路径。更多细节见 `yaml配置说明 <https://gitee.com/mindspore/federated/blob/r0.1/docs/api/api_python/horizontal/federated_server_yaml.md>`_。
        - **model** (nn.Cell) - 一个用于联邦训练的模型。
        - **sync_frequency** (int) - 联邦学习中的参数同步频率。若 `dataset_sink_mode` 设置为False，表示两个相邻同步操作之间的step数量。此时，若 `sync_type` 设置为"fixed"，其为固定的step数量。若 `sync_type` 设置为"adaptive"，其为动态同步频率的初始值。需要注意在数据下沉模式中，该参数的功能会改变。若 `dataset_sink_mode` 设置为True，且 `sink_size` 设置为一个非正数，同步操作将每间隔 `sync_frequency` 个epoch执行一次。若 `dataset_sink_mode` 设置为True，且 `sink_size` 设置为一个正数，同步操作将每间隔 `sink_size` * `sync_frequency` 个step执行一次。 `dataset_sink_mode` 和 `sink_size` 由用户在 `mindspore.train.Model` 中设置。
        - **http_server_address** (str) - 用于通信的http服务器地址。默认值：“”。
        - **data_size** (int) - 需要向worker报告的数据量。默认值：1。
        - **sync_type** (str) - 采用同步策略类型的参数。支持["fixed", "adaptive"]。默认值："fixed"。

          - fixed：参数的同步频率是固定的。
          - adaptive：参数的同步频率是自适应变化的。

        - **run_distribute** (bool) - 是否开启分布式训练。默认值：False。
        - **ssl_config** (Union(None, SSLConfig)) - ssl配置项。默认值：None。
        - **min_consistent_rate** (float) - 最小一致性比率阈值，该值越大同步频率提升难度越大。取值范围：大于等于0.0。默认值：1.1。
        - **min_consistent_rate_at_round** (int) - 最小一致性比率阈值的轮数，该值越大同步频率提升难度越大。取值范围：大于等于0。默认值：0。
        - **ema_alpha** (float) - 梯度一致性平滑系数，该值越小越会根据当前轮次的梯度分叉情况来判断频率是否需要改变，反之则会更加根据历史梯度分叉情况来判断。取值范围：(0.0, 1.0)。默认值：0.5。
        - **observation_window_size** (int) - 观察时间窗的轮数，该值越大同步频率减小难度越大。取值范围：大于0。默认值：5。
        - **frequency_increase_ratio** (int) - 频率提升幅度，该值越大频率提升幅度越大。取值范围：大于0。默认值：2。
        - **unchanged_round** (int) - 频率不发生变化的轮数，在前 `unchanged_round` 个轮次，频率不会发生变化。取值范围：大于等于0。默认值：0。
