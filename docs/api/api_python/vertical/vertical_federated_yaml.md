# 模型训练yaml详细配置项

MindSpore-Federated纵向联邦学习框架采用yaml配置文件，配置纵向联邦学习模型的训练与推理流程，包括网络、优化器、算子等模块的输入/输出和超参数信息。yaml配置文件的详细信息参见下表：

| 功能分类             | 配置参数                          | 参数类型                   | 取值范围                        | 是否必选 |
|------------------|-------------------------------|------------------------|-----------------------------|------|
| role             | role                          | str                    | 'leader' or 'follower'      | 是    |
| model            | train_net                     | dict                   |                             | 是    |
|                  | train_net.name                | str                    |                             | 否    |
|                  | train_net.inputs              | list                   |                             | 是    |
|                  | train_net.inputs.name         | str                    |                             | 是    |
|                  | train_net.inputs.source       | str                    | 'remote' or 'local'         | 是    |
|                  | train_net.inputs.compress_type | str                   | 'min_max' or 'bit_pack' or 'no_compress'        | 否    |
|                  | train_net.inputs.bit_num      | int                    | [1, 8]                      | 否    |
|                  | train_net.outputs             | list                   |                             | 是    |
|                  | train_net.outputs.name        | str                    |                             | 是    |
|                  | train_net.outputs.destination | str                    | 'remote' or 'local'         | 是    |
|                  | train_net.outputs.compress_type | str                  | 'min_max' or 'bit_pack' or 'no_compress'        | 否    |
|                  | train_net.outputs.bit_num     | int                    | [1, 8]                      | 否    |
|                  | eval_net                      | dict                   |                             | 是    |
|                  | eval_net.name                 | str                    |                             | 否    |
|                  | eval_net.inputs               | list                   |                             | 是    |
|                  | eval_net.inputs.name          | str                    |                             | 是    |
|                  | eval_net.inputs.source        | str                    | 'remote' or 'local'         | 是    |
|                  | eval_net.inputs.compress_type | str                   | 'min_max' or 'bit_pack' or 'no_compress'        | 否    |
|                  | eval_net.inputs.bit_num       | int                    | [1, 8]                      | 否    |
|                  | eval_net.outputs              | list                   |                             | 是    |
|                  | eval_net.output.name          | str                    |                             | 是    |
|                  | eval_net.output.destination   | str                    | 'remote' or 'local'         | 是    |
|                  | eval_net.outputs.compress_type | str                   | 'min_max' or 'bit_pack' or 'no_compress'        | 否    |
|                  | eval_net.outputs.bit_num      | int                    | [1, 8]                      | 否    |
|                  | eval_net.gt                   | str                    |                             | 否    |
| opts             | type                          | str                    | mindspore.nn.optim内定义的优化器名称 | 是    |
|                  | grads                         | list                   |                             | 是    |
|                  | grads.inputs                  | list                   |                             | 是    |
|                  | grads.inputs.name             | str                    |                             | 是    |
|                  | grads.output                  | dict                   |                             | 是    |
|                  | grads.output.name             | str                    |                             | 是    |
|                  | grads.params                  | list                   |                             | 否    |
|                  | grads.params.name             | str                    |                             | 否    |
|                  | grads.sens                    | union(float, int, str) |                             | 否    |
|                  | params                        | list                   |                             | 否    |
|                  | params.name                   | str                    |                             | 否    |
|                  | hyper_parameters              | dict                   |                             | 否    |
| grad_scalers     | inputs                        | list                   |                             | 否    |
|                  | inputs.name                   | str                    |                             | 否    |
|                  | output                        | dict                   |                             | 否    |
|                  | output.name                   | str                    |                             | 否    |
|                  | sens                          | union(float, int, str) |                             | 否    |
| dataset          | name                          | str                    |                             | 否    |
|                  | features                      | list                   |                             | 否    |
|                  | labels                        | list                   |                             | 否    |
| hyper_parameters | epochs                        | int                    |                             | 否    |
|                  | batch_size                    | int                    |                             | 否    |
|                  | is_eval                       | bool                   |                             | 否    |
| privacy          | label_dp                      | dict                   |                             | 否    |
|                  | label_dp.eps                  | float                  |                             | 否    |
| ckpt_path        |                               | str                    |                             | 否    |

其中：

- **role** (str) -  联邦学习参与方角色，必须是 ``"leader"`` 或 ``"follower"``。默认值：``""``。
- **train_net** (dict) - 描述训练网络输入、输出等信息的数据结构。默认值：``""``。
- **train_net.name** (str) - 训练网络名称标识符。默认值：``""``。
- **train_net.inputs** (list) - 训练网络输入Tensor列表，每个元素均为描述一个输入Tensor的字典。元素的排列顺序和名称，必须与MindSpore建模的训练网络（nn.Cell）construct方法的输入Tensor顺序和名称保持一致。默认值：``[]``。
- **train_net.inputs.name** (str) - 训练网络输入Tensor名称，必须与MindSpore建模的训练网络（nn.Cell）的输入Tensor名称保持一致。默认值：``""``。
- **train_net.inputs.source**(str) - 训练网络输入Tensor的数据来源，必须是 ``"remote"`` 或 ``"local"``，``"remote"`` 代表数据来源于其它参与方的网络传输，``"local"`` 代表数据来源于本地。默认值： ``"local"``。
- **train_net.inputs.compress_type**(str) - 压缩类型，必须是 ``"min_max"`` 或 ``"bit_pack"`` 或 ``"no_compress"``，``"min_max"`` 代表采用最小最大量化通信压缩方法，``"bit_pack"`` 代表采用比特打包通信压缩方法，``"no_compress"`` 代表不采用通信压缩方法。
- **train_net.inputs.bit_num**(int) - 通信压缩算法中的比特数。
- **train_net.outputs**  - (list) - 训练网络输出Tensor列表，每个元素均为描述一个输出Tensor的字典。元素的排列顺序和名称，必须与MindSpore建模的训练网络（nn.Cell）的construct方法的返回值Tensor顺序和名称保持一致。默认值：``[]``。
- **train_net.outputs.name** (str) - 训练网络输出Tensor名称，必须与MindSpore建模的训练网络（nn.Cell）的输出Tensor名称保持一致。默认值：``""``。
- **train_net.outputs.destination**(str) - 训练网络输出Tensor的数据去向，必须是 ``"remote"`` 或 ``"local"``，``"remote"`` 代表数据将通过网络传输给其它参与方，``"local"`` 代表数据本地使用，不进行网络传输。默认值： ``"local"``。
- **train_net.outputs.compress_type**(str) - 压缩类型，必须是 ``"min_max"`` 或 ``"bit_pack"或"no_compress"``，``"min_max"``代表采用最小最大量化通信压缩方法，``"bit_pack"`` 代表采用比特打包通信压缩方法，``"no_compress"`` 代表不采用通信压缩方法。
- **train_net.outputs.bit_num**(int) - 通信压缩算法中的比特数。
- **eval_net** (dict) - 描述评估网络输入、输出等信息的数据结构。默认值：``""``。
- **eval_net.name** (str) - 评估网络名称标识符。默认值：``""``。
- **eval_net.inputs** (list) - 评估网络输入Tensor列表，每个元素均为描述一个输入Tensor的字典。元素的排列顺序和名称，必须与MindSpore建模的评估网络（nn.Cell）construct方法的输入Tensor顺序和名称保持一致。默认值：``[]``。
- **eval_net.inputs.name** (str) - 评估网络输入Tensor名称，必须与MindSpore建模的训练网络（nn.Cell）的输入Tensor名称保持一致。默认值：``""``。
- **eval_net.inputs.source**(str) - 评估网络输入Tensor的数据来源，必须是 ``"remote"`` 或 ``"local"``，``"remote"`` 代表数据来源于其它参与方的网络传输，``"local"`` 代表数据来源于本地。默认值： ``"local"``。
- **eval_net.inputs.compress_type**(str) - 压缩类型，必须是 ``"min_max"`` 或 ``"bit_pack"`` 或 ``"no_compress"``，``"min_max"`` 代表采用最小最大量化通信压缩方法，``"bit_pack"`` 代表采用比特打包通信压缩方法，``"no_compress"`` 代表不采用通信压缩方法。
- **eval_net.inputs.bit_num**(int) - 通信压缩算法中的比特数。
- **eval_net.outputs**  - (list) - 评估网络输出Tensor列表，每个元素均为描述一个输出Tensor的字典。元素的排列顺序和名称，必须与MindSpore建模的评估网络（nn.Cell）的construct方法的返回值Tensor顺序和名称保持一致。默认值：``[]``。
- **eval_net.outputs.name** (str) - 评估网络输出Tensor名称，必须与MindSpore建模的评估网络（nn.Cell）的输出Tensor名称保持一致。默认值：``""``。
- **eval_net.outputs.destination**(str) - 评估网络输出Tensor的数据去向，必须是 ``"remote"`` 或 ``"local"``，``"remote"`` 代表数据将通过网络传输给其它参与方，``"local"`` 代表数据本地使用，不进行网络传输。默认值： ``"local"``。
- **eval_net.outputs.compress_type**(str) - 压缩类型，必须是 ``"min_max"`` 或 ``"bit_pack"`` 或 ``"no_compress"``，``"min_max"`` 代表采用最小最大量化通信压缩方法，``"bit_pack"`` 代表采用比特打包通信压缩方法，``"no_compress"`` 代表不采用通信压缩方法。默认值： ``"min_max"``。
- **eval_net.outputs.bit_num**(int) - 通信压缩算法中的比特数。
- **eval_net.gt**(str) - 评估网络输出对应的ground truth标签名称。默认值： ``""``。
- **type** (str) - 优化器类型，需采用mindspore.nn.optim内定义的优化器，如 ``"Adam"``，参考[优化器](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html#%E4%BC%98%E5%8C%96%E5%99%A8)。默认值： ``""``。
- **grads** (list) - 优化器关联的GradOperation列表，每个元素均为描述一个GradOperation算子的字典。默认值：``[]``。
- **grads.inputs** (list) - GradOperation算子的输入Tensor列表，每个元素均为描述一个输入Tensor的字典。默认值：``[]``。
- **grads.inputs.name** (str) - GradOperation算子的输入Tensor名称。默认值： ``""``。
- **grads.output** (dict) - 描述GradOperation算子对应的网络输出Tensor的字典。默认值：``{}``。
- **grads.output.name** (str) - GradOperation算子对应的网络输出Tensor名称。默认值： ``""``。
- **grads.params** (list) - GradOperation算子计算梯度值的训练网络参数列表，每个元素对应一个网络参数名称。如果为空，则将计算关联优化器所更新参数的梯度值。默认值：``[]``。
- **grads.params.name** (str) - GradOperation算子计算梯度值的训练网络参数名称。默认值： ``""``。
- **grads.sens** (union(float, int, str)) - GradOperation算子计算网络参数梯度的加权系数，对应GradOperation算子的"灵敏度"（参考[mindspore.ops.GradOperation](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.GradOperation.html?highlight=gradoperation)）。如果是float或int类型，则采用常量作为加权系数；如果是str类型，则从其它参与方经网络传输的加权系数中，解析名称与其对应的Tensor作为加权系数。默认值： ``""``。
- **params** (list) - 优化器根据梯度计算结果，更新的训练网络参数列表，每个元素对应一个网络参数名称。如果为空，则优化器将更新训练为例的所有可训练参数。默认值：``[]``。
- **params.name** (str) - 优化器更新的训练网络的参数名称。默认值： ``""``。
- **hyper_parameters** (dict) - 优化器超参数字典，参考type所指定的MindSpore优化器算子的超参数。默认值：``{}``。
- **grad_scalers.inputs** (list) - 用于计算梯度加权系数的GradOperation算子的输入Tensor列表，每个元素均为描述一个输入Tensor的字典。默认值：``[]``。
- **grad_scalers.inputs.name** (str) - 用于计算梯度加权系数的GradOperation算子的输入Tensor名称。默认值： ``""``。
- **grad_scalers.output** (list) - 描述用于计算梯度加权系数的GradOperation算子对应的网络输出Tensor的字典.默认值：``{}``。
- **grad_scalers.output.name** (str) - 描述用于计算梯度加权系数的GradOperation算子对应的网络输出Tensor名称。默认值： ``""``。
- **grad_scalers.sens** (str) - 描述用于计算梯度加权系数的GradOperation算子计算网络参数梯度的加权系数，对应GradOperation算子的"灵敏度"（参考[mindspore.ops.GradOperation](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.GradOperation.html?highlight=gradoperation)）。如果是float或int类型，则采用常量作为加权系数；如果是str类型，则从其它参与方经网络传输的数据中，解析名称与其对应的Tensor作为加权系数。默认值： ``""``。
- **dataset.name** (str) - 数据集名称。默认值： ``""``。
- **dataset.features** (list) - 数据集特征列表，每个元素均为一个str类型的特征名称。默认值：``[]``。
- **dataset.labels** (list) - 数据集标签列表，每个元素均为一个str类型的标签名称。默认值：``[]``。
- **epochs** (int) - 训练的epoch数。默认值：``1``。
- **batch_size** (int) - 训练的数据batch size。默认值：``1``。
- **is_eval** (bool) - 训练完成后是否执行评估。默认值：``False``。
- **label_dp** (dict) - 差分隐私机制的配置参数。默认值：``{}``。
- **label_dp.eps** (float) - 差分隐私机制的eps参数。默认值：``1.0``。
- **ckpt_path** (str) - 保存训练网络checkpoint文件的路径。默认值：``"./checkpoints"``。

以本项目所提供的[纵向联邦学习模型训练 - Wide&Deep推荐应用](https://gitee.com/mindspore/federated/tree/master/example/splitnn_criteo)为例，其基于Wide&Deep模型和Criteo数据集，进行纵向联邦学习模型训练，其纵向联邦Leader参与方的yaml如下：

```yaml
role: leader
model: # define the net of vFL party
  train_net:
    name: leader_loss_net
    inputs:
      - name: id_hldr
        source: local
      - name: wt_hldr
        source: local
      - name: wide_embedding
        source: remote
        compress_type: min_max
        bit_num: 6
      - name: deep_embedding
        source: remote
        compress_type: min_max
        bit_num: 6
      - name: label
        source: local
    outputs:
      - name: out
        destination: local
      - name: wide_loss
        destination: local
      - name: deep_loss
        destination: local
  eval_net:
    name: leader_eval_net
    inputs:
      - name: id_hldr
        source: local
      - name: wt_hldr
        source: local
      - name: wide_embedding
        source: remote
        compress_type: min_max
        bit_num: 6
      - name: deep_embedding
        source: remote
        compress_type: min_max
        bit_num: 6
    outputs:
      - name: logits
        destination: local
      - name: pred_probs
        destination: local
    gt: label
opts: # define ms optimizer
  - type: FTRL
    grads: # define ms grad operations
      - inputs:
          - name: id_hldr
          - name: wt_hldr
          - name: wide_embedding
          - name: deep_embedding
          - name: label
        output:
          name: wide_loss
        sens: 1024.0
        # if not specify params, inherit params of optimizer
    params:  # if not specify params, process all trainable params
      - name: wide
    hyper_parameters:
      learning_rate: 5.e-2
      l1: 1.e-8
      l2: 1.e-8
      initial_accum: 1.0
      loss_scale: 1024.0
  - type: Adam
    grads:
      - inputs:
          - name: id_hldr
          - name: wt_hldr
          - name: wide_embedding
          - name: deep_embedding
          - name: label
        output:
          name: deep_loss
        sens: 1024.0
    params:
      - name: deep
      - name: dense
    hyper_parameters:
      learning_rate: 3.5e-4
      eps: 1.e-8
      loss_scale: 1024.0
grad_scalers: # define the grad scale calculator
  - inputs:
      - name: wide_embedding
      - name: deep_embedding
    output:
      name: wide_loss
    sens: 1024.0
  - inputs:
      - name: wide_embedding
      - name: deep_embedding
    output:
      name: deep_loss
    sens: 1024.0
dataset:
  name: criteo
  features:
    - id_hldr
    - wt_hldr
  labels:
    - ctr
hyper_parameters:
  epochs: 20
  batch_size: 16000
  is_eval: True
ckpt_path: './checkpoints'
```
