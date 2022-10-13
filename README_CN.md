# MindSpore Federated

[View English](./README.md)

## 概述

MindSpore Federated是一款开源联邦学习框架，支持面向千万级无状态终端设备的商用化部署，可在用户数据不出本地的前提下，使能全场景智能应用。

联邦学习是一种加密的分布式机器学习技术，其支持机器学习的各参与方在不直接共享本地数据的前提下，共建AI模型。MindSpore Federated目前优先专注于参与方数量规模较大的横向联邦学习应用场景。

<img src="docs/api/api_python/architecture.png" alt="MindSpore Architecture" width="600"/>

## 安装

MindSpore Federated的安装可以采用pip安装或者源码编译安装两种方式。

### pip安装

使用pip命令安装，请从[MindSpore Federated下载页面](https://www.mindspore.cn/versions)下载并安装whl包。

 ```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/Federated/{arch}/mindspore_federated-{version}-{python_version}-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - `{version}`表示MindSpore Federated版本号，例如下载1.1.0版本MindSpore Federated时，`{version}`应写为1.1.0。
> - `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARM架构64位，则写为`aarch64`。
> - `{python_version}`表示用户的Python版本，Python版本为3.7.5时，`{python_version}`应写为`cp37-cp37m`。Python版本为3.9.0时，则写为`cp39-cp39`。请和当前安装的MindSpore Federated使用的Python环境保持一致。

### 源码编译安装

通过[源码](https://gitee.com/mindspore/federated)编译安装。

```shell
git clone https://gitee.com/mindspore/federated.git -b master
cd federated
bash build.sh
```

对于`bash build.sh`，可通过例如`-jn`选项，例如`-j16`，加速编译；可通过`-S on`选项，从gitee而不是github下载第三方依赖。

编译完成后，在`build/package/`目录下找到Federated的whl安装包进行安装：

```python
pip install mindspore_federated-{version}-{python_version}-linux_{arch}.whl
```

## 验证是否成功安装

执行以下命令，验证安装结果。导入Python模块不报错即安装成功：

```python
from mindspore_federated import FLServerJob
```

## 运行样例

后续独立文档。

### 环境准备

#### 安装和启动Redis服务器

联邦学习默认依赖[Redis服务器](https://redis.io/)作为缓存数据中间件，运行联邦学习业务，需要安装和运行Redis服务器。

安装Redis服务器：

```shell
sudo apt-get install redis
```

运行Redis服务器：

```shell
redis-server --port 2345 --save ""
```

#### 安装MindSpore

Worker运行环境：混合模式和云云联邦场景，根据Worker依赖的硬件环境安装相应的[MindSpore](https://www.mindspore.cn/install)包。

Server和Scheduler运行环境：只需要安装CPU版本的MindSpore whl，其他硬件版本的MindSpore whl包也能满足需要。

### MindSpore版本依赖关系

由于MindSpore Federated与MindSpore有依赖关系，请按照根据下表中所指示的对应关系，在[MindSpore下载页面](https://www.mindspore.cn/versions)下载并安装对应的whl包。

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{MindSpore-Version}/MindSpore/cpu/ubuntu_x86/mindspore-{MindSpore-Version}-cp37-cp37m-linux_x86_64.whl
```

| MindSpore Federated Version |                          Branch                          | MindSpore version |
|:---------------------------:|:--------------------------------------------------------:|:-----------------:|
|            0.1.0            | [r0.1](https://gitee.com/mindspore/federated/tree/r0.1/) |       1.7.0       |
|            0.1.0            | [r0.1](https://gitee.com/mindspore/federated/tree/r0.1/) |       1.8.0       |
|            0.1.0            | [r0.1](https://gitee.com/mindspore/federated/tree/r0.1/) |       1.9.0       |

#### 非混合模式，端云联邦

1、样例路径：

```shell
cd tests/st/cross_device_cloud
```

2、据实际运行需要修改Yaml配置文件：`default_yaml_config.yaml`

3、运行Server，默认启动1个Server，HTTP服务器地址为`127.0.0.1:6666`

```shell
python run_server.py
```

可通过指定`${http_server_address}`设置HTTP Server的IP+端口号，默认为6666，`${server_num}`为启动的Server个数:

```shell
python run_server.py --http_server_address=${http_server_address} --local_server_num=${server_num}
```

注意可以通过输入checkpoint的路径(指定`checkpoint_dir`)与构造神经网络的方式传入`feature_map`,请参考cross_device_femnist目录下的run_cloud.py文件。

4、启动Scheduler，管理面地址默认为`127.0.0.1:11202`

```shell
python run_sched.py
```

可通过额外指定`scheduler_manage_address`设定管理面地址，其中`${scheduler_manage_address}`为Scheduler管理面HTTP服务器的地址。

```shell
python run_sched.py --scheduler_manage_address=${scheduler_manage_address}
```

#### 混合联邦模式

1、样例路径：

```shell
cd tests/st/hybrid_cloud
```

2、据实际运行需要修改Yaml配置文件：`default_yaml_config.yaml`

3、运行Server，默认启动1个Server，HTTP服务器地址为`127.0.0.1:6666`

```shell
python run_hybrid_train_server.py
```

可通过指定`${http_server_address}`设置HTTP Server的IP+端口号，默认为6666，`${server_num}`为启动的Server个数:

```shell
python run_hybrid_train_server.py --http_server_address=${http_server_address} --local_server_num=${server_num}
```

4、启动Scheduler，管理面地址默认为`127.0.0.1:11202`

```shell
python run_hybrid_train_sched.py
```

可通过额外指定`scheduler_manage_address`设定管理面地址，其中`${scheduler_manage_address}`为Scheduler管理面HTTP服务器的地址。

```shell
python run_hybrid_train_sched.py --scheduler_manage_address=${scheduler_manage_address}
```

5、运行Worker，运行于云侧启动有监督训练，默认启动1个Worker，HTTP服务器地址为`127.0.0.1:6666`

```shell
python run_hybrid_train_worker.py
```

可通过额外，其中`${dataset_path}`为训练集数据的路径。

```shell
python run_hybrid_train_worker.py  --dataset_path=${dataset_path}
```

#### 云云联邦模式

1、样例路径：

```shell
cd tests/st/cross_silo_femnist
```

2、据实际运行需要修改Yaml配置文件：`default_yaml_config.yaml`

3、运行Server，默认启动1个Server，HTTP服务器地址为`127.0.0.1:6666`

```shell
python run_cross_silo_femnist_server.py
```

可通过指定`${http_server_address}`设置HTTP Server的IP+端口号，默认为6666，`${server_num}`为启动的Server个数:

```shell
python run_cross_silo_femnist_server.py --http_server_address=${http_server_address} --local_server_num=${server_num}
```

4、启动Scheduler，管理面地址默认为`127.0.0.1:11202`

```shell
python run_cross_silo_femnist_sched.py
```

可通过额外指定`scheduler_manage_address`设定管理面地址，其中`${scheduler_manage_address}`为Scheduler管理面HTTP服务器的地址。

```shell
python run_cross_silo_femnist_sched.py --scheduler_manage_address=${scheduler_manage_address}
```

5、运行Worker，运行于端侧启动有监督训练，多个worker之间参与联合建模，HTTP服务器地址为`127.0.0.1:6666`

```shell
python run_cross_silo_femnist_worker.py
```

可通过额外，其中`${dataset_path}`为训练集数据的路径。

```shell
python run_cross_silo_femnist_worker.py  --dataset_path=${dataset_path}
```

最后：启动客户端Python模拟

```shell
bash run_smlt.sh 4 http 127.0.0.1:6666
```

其中`4`为启动模拟客户端的数量，`http`为启动访问的方式，`127.0.0.1:6666`为服务器server集群的监听地址。

#### 启动ssl通信加密访问模式

1、样例路径：

```shell
cd tests/st/cross_device_cloud
```

2、证书生成样例请运行/tests/ut/python/generate_certs.sh

3、据实际运行需要修改Yaml配置文件：`default_yaml_config.yaml`，需要设置enable_ssl为true，设置cacert_filename(path/to/ca.crt), cert_filename(/path/to/client.crt), private_key_filename(/path/to/clientkey.pem)

4、以ssl方式运行Redis服务器：

```shell
redis-server --port 0 --tls-port 2345 --tls-cert-file /path/to/server.crt --tls-key-file /path/to/serverkey.pem --tls-ca-cert-file /path/to/ca.crt --save ""
```

5、运行Server，默认启动1个Server，HTTP服务器地址为`127.0.0.1:6666`, 需要在run_server.py设置秘钥参数对象SSLConfig("server_password", "client_password"), 表示客户端证书秘钥与服务器证书秘钥的密码。

```shell
python run_server.py
```

可通过指定`${http_server_address}`设置HTTP Server的IP+端口号，默认为6666，`${server_num}`为启动的Server个数:

```shell
python run_server.py --http_server_address=${http_server_address} --local_server_num=${server_num}
```

注意可以通过输入checkpoint的路径(指定`checkpoint_dir`)与构造神经网络的方式传入`feature_map`,请参考cross_device_femnist目录下的run_cloud.py文件。

6、启动Scheduler，管理面地址默认为`127.0.0.1:11202`

```shell
python run_sched.py
```

7、可通过额外指定`scheduler_manage_address`设定管理面地址，其中`${scheduler_manage_address}`为Scheduler管理面HTTP服务器的地址。

```shell
python run_sched.py --scheduler_manage_address=${scheduler_manage_address}
```

最后：启动客户端Python模拟

```shell
bash run_smlt.sh 4 https 127.0.0.1:6666
```

其中`4`为启动模拟客户端的数量，`https`为启动访问的方式，`127.0.0.1:6666`为服务器server集群的监听地址。

###

## 快速入门

## 文档

### 开发者教程

有关安装指南、教程和API的更多详细信息，请参阅[用户文档](https://www.mindspore.cn/federated/docs/zh-CN/master/deploy_federated_server.html)。

## 社区

### 治理

查看MindSpore如何进行[开放治理](https://gitee.com/mindspore/community/blob/master/governance.md)。

### 交流

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w) 开发者交流平台。

## 贡献

欢迎参与贡献。

## 版本说明

版本说明请参阅[RELEASE](RELEASE.md)。

## 许可证

[Apache License 2.0](LICENSE)
