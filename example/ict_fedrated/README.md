## DeepFM概述

要想在推荐系统中实现最大点击率，学习用户行为背后复杂的特性交互十分重要。虽然已在这一领域取得很大进展，但高阶交互和低阶交互的方法差异明显，亟需专业的特征工程。本论文中,我们将会展示高阶和低阶交互的端到端学习模型的推导。本论文提出的模型DeepFM，结合了推荐系统中因子分解机和新神经网络架构中的深度特征学习。

[论文](https://arxiv.org/abs/1703.04247):  Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

## 模型架构

DeepFM由两部分组成。FM部分是一个因子分解机，用于学习推荐的特征交互；深度学习部分是一个前馈神经网络，用于学习高阶特征交互。
FM和深度学习部分拥有相同的输入原样特征向量，让DeepFM能从输入原样特征中同时学习低阶和高阶特征交互。

## 数据集

- [Criteo Kaggle Display Advertising Challenge Dataset](http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz)

- 数据集处理详见：https://gitee.com/mindspore/models/tree/master/official/recommend/deepfm

## 脚本说明

## 脚本和样例代码

```bash
.
├── FedHuaiwei-baseline						//对比方案代码
│   └── deepfm
│       ├── client.py							//客户端侧代码
│       ├── data								//数据集文件
│       ├── default_config.yaml				//默认配置文件
│       ├── loss.log							//日志文件
│       ├── requirements.txt					//运行所需环境
│       ├── server							//服务端侧文件
│       └── src								//模型文件
└── FedHuawei-new							//自研方案代码
    └── deepfm
        ├── client.py							//客户端侧代码
        ├── data								//数据集文件
        ├── default_config.yaml				//默认配置文件
        ├── loss.log							//日志文件
        ├── requirements.txt					//运行所需环境
        ├── server							//服务端侧文件
        └── src								//模型文件

```

## 脚本参数

在`default_config.yaml`中可以配置训练参数


```python
  optional arguments:
  -h, --help            show this help message and exit
  
  "data config"
    data_vocab_size: 184965
    train_num_of_parts: 21
    test_num_of_parts: 3
    batch_size: 1000 #0
    data_field_size: 39
    data_format: 1

  "model config"
    data_emb_dim: 80
    deep_layer_args: [[1024, 512, 256, 128], "relu"]
    init_args: [-0.01, 0.01]
    weight_bias_init: ['normal', 'normal']
    keep_prob: 0.9
    convert_dtype: True

  "train config"
    l2_coef: 0.00008 # 8e-5
    learning_rate: 0.0005 # 5e-4
    epsilon: 0.00000005 # 5e-8
    loss_scale: 1024.0
    loss_callback: True
    train_epochs: 1

 "fl train config"
    fl_mode: "fedavg" # the fl mechanism: fedavg,fedasync(only in baseline),mi,dfedasync(only in new)
    num_clients: 20 # total number of clients
    num_samples: 10 # num_samples each client has
    train_ratio: 0.8 #the percent of training samples for each client
    max_round: 3 #maximum fl round
    num_client_per_round: 4  # number of clients selected each round
    cal_staleness: 1 #for fedasync mode of cal staleness
    agg_client_per_round: 1 #number of clients aggregated for  fedasync(only baseline)
    sigma: 10.0 # for fedasync control the variance of distribution of training time

  "train.py 'CTR Prediction'"
    dataset_path: "./data/mindrecord"
  ```



## 启动脚本

```bash
python train.py 
```  

