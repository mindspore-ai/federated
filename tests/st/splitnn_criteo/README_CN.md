# MindSpore Federated纵向联邦学习样例

## 概述

MindSpore Federated提供基于拆分学习（Split Learning）的纵向联邦学习基础功能组件。本样例以Wide&Deep网络和Criteo数据集为例，提供了面向推荐任务的联邦学习训练样例。

<img src="../../../docs/splitnn_wide_and_deep.png" alt="MindSpore Federated纵向联邦学习样例" width="854"/>

如上图所示，该案例中，纵向联邦学习系统由Leader参与方和Follower参与方组成。其中，Leader参与方持有20×2维特征信息和标签信息，Follower参与方持有19×2维特征信息。Leader参与方和Follower参与方分别部署1组Wide&Deep网络，并通过交换embedding向量和梯度向量，在不泄露原始特征和标签信息的前提下，实现对网络模型的协同训练。

Wide&Deep网络原理特性的详细介绍，可参考[MindSpore ModelZoo - Wide&Deep - Wide&Deep概述](https://gitee.com/mindspore/models/blob/master/official/recommend/wide_and_deep/README_CN.md#widedeep%E6%A6%82%E8%BF%B0)及其[研究论文](https://arxiv.org/pdf/1606.07792.pdf)。

## 数据集准备

本样例基于Criteo数据集进行训练和测试，在运行样例前，需参考[MindSpore ModelZoo - Wide&Deep - 快速入门](https://gitee.com/mindspore/models/blob/master/official/recommend/wide_and_deep/README_CN.md#%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8)，对Criteo数据集进行预处理。

1. 克隆MindSpore ModelZoo代码。

```bash
git clone https://gitee.com/mindspore/models.git
cd models/official/recommend/wide_and_deep
```

2. 下载数据集。

```bash
mkdir -p data/origin_data && cd data/origin_data
wget http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
tar -zxvf criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
```

3. 使用此脚本预处理数据。预处理过程可能需要一小时，生成的MindRecord数据存放在data/mindrecord路径下。预处理过程内存消耗巨大，建议使用服务器。

```bash
python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
```

## 运行样例

本样例提供2个示例程序，均以Shell脚本拉起Python程序的形式运行。

1. run_vfl_train_local.sh：单线程示例程序，Leader参与方和Follower参与方在同一线程中训练，其以程序内变量的方式，直接传输embedding向量和梯度向量至另一参与方。

2. run_vfl_train_socket.sh：多线程示例程序，Leader参与方和Follower参与方分别运行一个训练线程，其分别将embedding向量和梯度向量封装为protobuf消息后，通过socket通信接口传输至另一参与方。

以run_vfl_train_local.sh为例，运行示例程序的步骤如下：

1. 参考[MindSpore官网指引](https://www.mindspore.cn/install)，安装MindSpore 1.8.1。

2. 采用安装MindSpore Federated所依赖Python库。

```bash
cd federated
python -m pip install -r requirements_test.txt
```

3. 拷贝[预处理](#数据集准备)后的Criteo数据集至本目录下。

```bash
cd tests/st/splitnn_criteo
cp -rf ${DATA_ROOT_PATH}/data/mindrecord/ ./
```

4. 运行示例程序启动脚本。

```bash
./run_vfl_train_local.sh
```

5. 查看训练日志`log_local_gpu.txt`。

```sh
INFO:root:epoch 0 step 100/2582 wide_loss: 0.528141 deep_loss: 0.528339
INFO:root:epoch 0 step 200/2582 wide_loss: 0.499408 deep_loss: 0.499410
INFO:root:epoch 0 step 300/2582 wide_loss: 0.477544 deep_loss: 0.477882
INFO:root:epoch 0 step 400/2582 wide_loss: 0.474377 deep_loss: 0.476771
INFO:root:epoch 0 step 500/2582 wide_loss: 0.472926 deep_loss: 0.475157
INFO:root:epoch 0 step 600/2582 wide_loss: 0.464844 deep_loss: 0.467011
INFO:root:epoch 0 step 700/2582 wide_loss: 0.464496 deep_loss: 0.466615
INFO:root:epoch 0 step 800/2582 wide_loss: 0.466895 deep_loss: 0.468971
INFO:root:epoch 0 step 900/2582 wide_loss: 0.463155 deep_loss: 0.465299
INFO:root:epoch 0 step 1000/2582 wide_loss: 0.457914 deep_loss: 0.460132
INFO:root:epoch 0 step 1100/2582 wide_loss: 0.453361 deep_loss: 0.455767
INFO:root:epoch 0 step 1200/2582 wide_loss: 0.457566 deep_loss: 0.459997
INFO:root:epoch 0 step 1300/2582 wide_loss: 0.460841 deep_loss: 0.463281
INFO:root:epoch 0 step 1400/2582 wide_loss: 0.460973 deep_loss: 0.463365
INFO:root:epoch 0 step 1500/2582 wide_loss: 0.459204 deep_loss: 0.461563
INFO:root:epoch 0 step 1600/2582 wide_loss: 0.456771 deep_loss: 0.459200
INFO:root:epoch 0 step 1700/2582 wide_loss: 0.458479 deep_loss: 0.460963
INFO:root:epoch 0 step 1800/2582 wide_loss: 0.449609 deep_loss: 0.452122
INFO:root:epoch 0 step 1900/2582 wide_loss: 0.451775 deep_loss: 0.454225
INFO:root:epoch 0 step 2000/2582 wide_loss: 0.460343 deep_loss: 0.462826
INFO:root:epoch 0 step 2100/2582 wide_loss: 0.456814 deep_loss: 0.459201
INFO:root:epoch 0 step 2200/2582 wide_loss: 0.452091 deep_loss: 0.454555
INFO:root:epoch 0 step 2300/2582 wide_loss: 0.461522 deep_loss: 0.464001
INFO:root:epoch 0 step 2400/2582 wide_loss: 0.442355 deep_loss: 0.444790
INFO:root:epoch 0 step 2500/2582 wide_loss: 0.450675 deep_loss: 0.453242
...
```
