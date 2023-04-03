# Yaml Configuration file for model training of vertical federated learning

MindSpore-Federated adopts a yaml file to configure the training and predicting processes of vertical federated learning. The yaml configuration file contains information on inputs/outputs and hyper-parameters of neural networks, optimizers, operators, and other modules. Details of the yaml configuration file are as follows:

| Classification   | Parameters                    | Type                   | Value Range                               | Required/Optional |
|------------------|-------------------------------|------------------------|-------------------------------------------|-------------------|
| role             | role                          | str                    | 'leader' or 'follower'                    | Required          |
| model            | train_net                     | dict                   |                                           | Required          |
|                  | train_net.name                | str                    |                                           | Optional          |
|                  | train_net.inputs              | list                   |                                           | Required          |
|                  | train_net.inputs.name         | str                    |                                           | Required          |
|                  | train_net.inputs.source       | str                    | 'remote' or 'local'                       | Required          |
|                  | train_net.inputs.compress_type | str                   | 'min_max' or 'bit_pack' or 'no_compress'  | Optional          |
|                  | train_net.inputs.bit_num      | int                    | [1, 8]                                    | Optional          |
|                  | train_net.outputs             | list                   |                                           | Required          |
|                  | train_net.outputs.name        | str                    |                                           | Required          |
|                  | train_net.outputs.destination | str                    | 'remote' or 'local'                       | Required          |
|                  | train_net.outputs.compress_type | str                  | 'min_max' or 'bit_pack' or 'no_compress'  | Optional          |
|                  | train_net.outputs.bit_num     | int                    | [1, 8]                                    | Optional          |
|                  | eval_net                      | dict                   |                                           | Required          |
|                  | eval_net.name                 | str                    |                                           | Optional          |
|                  | eval_net.inputs               | list                   |                                           | Required          |
|                  | eval_net.inputs.name          | str                    |                                           | Required          |
|                  | eval_net.inputs.source        | str                    | 'remote' or 'local'                       | Required          |
|                  | eval_net.inputs.compress_type | str                    | 'min_max' or 'bit_pack' or 'no_compress'  | Optional          |
|                  | eval_net.inputs.bit_num       | int                    | [1, 8]                                    | Optional          |
|                  | eval_net.outputs              | list                   |                                           | Required          |
|                  | eval_net.output.name          | str                    |                                           | Required          |
|                  | eval_net.output.destination   | str                    | 'remote' or 'local'                       | Required          |
|                  | eval_net.outputs.compress_type | str                   | 'min_max' or 'bit_pack' or 'no_compress'  | Optional          |
|                  | eval_net.outputs.bit_num      | int                    | [1, 8]                                    | Optional          |
|                  | eval_net.gt                   | str                    |                                           | Optional          |
| opts             | type                          | str                    | names of optimizers in mindspore.nn.optim | Required          |
|                  | grads                         | list                   |                                           | Required          |
|                  | grads.inputs                  | list                   |                                           | Required          |
|                  | grads.inputs.name             | str                    |                                           | Required          |
|                  | grads.output                  | dict                   |                                           | Required          |
|                  | grads.output.name             | str                    |                                           | Required          |
|                  | grads.params                  | list                   |                                           | Optional          |
|                  | grads.params.name             | str                    |                                           | Optional          |
|                  | grads.sens                    | union(float, int, str) |                                           | Optional          |
|                  | params                        | list                   |                                           | Optional          |
|                  | params.name                   | str                    |                                           | Optional          |
|                  | hyper_parameters              | dict                   |                                           | Optional          |
| grad_scalers     | inputs                        | list                   |                                           | Optional          |
|                  | inputs.name                   | str                    |                                           | Optional          |
|                  | output                        | dict                   |                                           | Optional          |
|                  | output.name                   | str                    |                                           | Optional          |
|                  | sens                          | union(float, int, str) |                                           | Optional          |
| dataset          | name                          | str                    |                                           | Optional          |
|                  | features                      | list                   |                                           | Optional          |
|                  | labels                        | list                   |                                           | Optional          |
| hyper_parameters | epochs                        | int                    |                                           | Optional          |
|                  | batch_size                    | int                    |                                           | Optional          |
|                  | is_eval                       | bool                   |                                           | Optional          |
| privacy          | label_dp                      | dict                   |                                           | Optional          |
|                  | label_dp.eps                  | float                  |                                           | Optional          |
| ckpt_path        |                               | str                    |                                           | Optional          |

Parameters:

- **role** (str) -  Role of federated learning party, shall be either "leader" or "follower". Default: "".
- **train_net** (dict) - Data structure describing information on the training network, including inputs, outputs, etc. Default: "".
- **train_net.name** (str) - Name of the training network. Default: "".
- **train_net.inputs** (list) - Input tensor list of the training network. Each item of the list is a dict describing an input tensor. The sequence and names of items shall be the same as the input variables of the "construct" function of the training network (derived from mindspore.nn.Cell). Default: [].
- **train_net.inputs.name** (str) - Name of an input tensor of the training network. Shall be the same as the corresponding input of the training network modeled with mindspore.nn.Cell. Default: "".
- **train_net.inputs.source**(str) - Source of an input tensor of the training network. Shall be either "remote" or "local". "remote" indicates that the input tensor is received from another party through network. "local" indicates that the input tensor is loaded locally. Default: "local".
- **train_net.inputs.compress_type**(str) - Compress type. Shall be either "min_max" or "bit_pack" or "no_compress". "min_max" indicates min max communication compress method is used. "bit_pack" indicates bit pack communication compress method is used. "no_compress" indicates communication compress method is not used.
- **train_net.inputs.bit_num**(int) - The bit number in communication compression.
- **train_net.outputs**  - (list) - Output tensor list of the training network. Each item of the list is a dict describing an output tensor. The sequence and names of items shall be the same as the returning values of the "construct" function of the training network (derived from mindspore.nn.Cell). Default: [].
- **train_net.outputs.name** (str) - Name of an output tensor of the training network. Shall be the same as the corresponding output of the training network modeled with mindspore.nn.Cell. Default: "".
- **train_net.outputs.destination**(str) - Indicating where the output tensor is going. Shall be either "remote" or "local". "remote" indicates that the output tensor will be sending to another party through network. "local" indicates that the output tensor will be used locally. Default:  "local".
- **train_net.outputs.compress_type**(str) - Compress type. Shall be either "min_max" or "bit_pack" or "no_compress". "min_max" indicates min max communication compress method is used. "bit_pack" indicates bit pack communication compress method is used. "no_compress" indicates communication compress method is used.
- **train_net.outputs.bit_num**(int) - The bit number in communication compression.
- **eval_net** (dict) - Data structure describing information on the evaluation network, including inputs, outputs, etc. Default: "".
- **eval_net.name** (str) - Name of the evaluation network. Default: "".
- **eval_net.inputs** (list) - Input tensor list of the evaluation network. Each item of the list is a dict describing an input tensor. The sequence and names of items shall be the same as the input variables of the "construct" function of the evaluation network (derived from mindspore.nn.Cell). Default: [].
- **eval_net.inputs.name** (str) - Name of an input tensor of the evaluation network. Shall be the same as the corresponding input of the evaluation network modeled with mindspore.nn.Cell. Default: "".
- **eval_net.inputs.source**(str) - Source of an input tensor of the evaluation network. Shall be either "remote" or "local". "remote" indicates that the input tensor is received from another party through network. "local" indicates that the input tensor is loaded locally. Default: "local".
- **eval_net.inputs.compress_type**(str) - Compress type. Shall be either "min_max" or "bit_pack" or "no_compress". "min_max" indicates min max communication compress method is used. "bit_pack" indicates bit pack communication compress method is used. "no_compress" indicates communication compress method is used.
- **eval_net.inputs.bit_num**(int) - The bit number in communication compression.
- **eval_net.outputs**  - (list) - Output tensor list of the evaluation network. Each item of the list is a dict describing an output tensor. The sequence and names of items shall be the same as the returning values of the "construct" function of the evaluation network (derived from mindspore.nn.Cell). Default: [].
- **eval_net.outputs.name** (str) - Name of an output tensor of the evaluation network. Shall be the same as the corresponding output of the evaluation network modeled with mindspore.nn.Cell. Default: "".
- **eval_net.outputs.destination**(str) - Indicating where the output tensor is going. Shall be either "remote" or "local". "remote" indicates that the output tensor will be sending to another party through network. "local" indicates that the output tensor will be used locally. Default:  "local".
- **eval_net.outputs.compress_type**(str) - Compress type. Shall be either "min_max" or "bit_pack" or "no_compress". "min_max" indicates min max communication compress method is used. "bit_pack" indicates bit pack communication compress method is used. "no_compress" indicates communication compress method is used.
- **eval_net.outputs.bit_num**(int) - The bit number in communication compression.
- **eval_net.gt**(str) - Name of ground truth which will be compared with the prediction of the evaluation network. Default:  "".
- **type** (str) - Type of optimizer. Shall be the name of an optimizer in mindspore.nn.optim, like "Adam". Please refer to [Optimizer](https://mindspore.cn/docs/en/r2.0/api_python/mindspore.nn.html#optimizer). Default: "".
- **grads** (list) - List of GradOperation operators related to the optimizer. Each item of the list is a dict describing a GradOperation operator. Default: [].
- **grads.inputs** (list) - List of input tensors related to the GradOperation operator. Each item of the list is a dict describing an input tensor. Default: [].
- **grads.inputs.name** (str) - Name of an input tensor related to the GradOperation operator. Default: "".
- **grads.output** (dict) - Output tensor related to the GradOperation operator. Default: {}.
- **grads.output.name** (str) - Name of the output tensor related to the GradOperation operator. Default: "".
- **grads.params** (list) - List of weights of the training network, gradients of which will be calculated by the GradOperation operator. Each item is a name of weights. If the list is empty, gradients of weights defined in opts.params will be calculated. Default: [].
- **grads.params.name** (str) - Name of weights of the training network, gradients of which will be calculated by the GradOperation operator. Default: "".
- **grads.sens** (union(float, int, str)) - Sensitivity (gradient with respect to output) of the GradOperation operator used for calculating the gradients of weights of the training network. (Please refer to [mindspore.ops.GradOperation](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.GradOperation.html)). If it is a float or int value, the sensitivity will be set to a constant tensor. If it is a str value, the sensitivity will be parsed from variable data received from other parties. Default: "".
- **params** (list) - List of weights of the training network will be updated by the optimizer. Each item is a name of weights. If the list is empty, the optimizer will update all trainable weights of the training network. Default: [].
- **params.name** (str) - Name of weights of the training network will be updated by the optimizer. Default: "".
- **hyper_parameters** (dict) - Hyper-parameters of the optimizer. Please refer to the API of the optimizer operator. Default: {}.
- **grad_scalers.inputs** (list) - List of input tensors related to the GradOperation operator used for calculating the sensitivity. Each item is a dict describing an input tensor. Default: [].
- **grad_scalers.inputs.name** (str) - Name of an input tensor related to the GradOperation operator used for calculating the sensitivity. Default: "".
- **grad_scalers.output** (list) - Dict describing the output tensor related to the GradOperation operator used for calculating the sensitivity. Default: {}.
- **grad_scalers.output.name** (str) - Name of the output tensor related to the GradOperation operator used for calculating the sensitivity. Default: "".
- **grad_scalers.sens** (str) - Sensitivity (gradient with respect to output) of the GradOperation operator used for calculating the sensitivity. (Please refer to [mindspore.ops.GradOperation](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.GradOperation.html)). If it is of type float or int, the sensitivity will be set to a constant tensor. If it is of type str, the sensitivity will be parsed from variable data received from other parties. Default: "".
- **dataset.name** (str) - Name of dataset. Default: "".
- **dataset.features** (list) - Feature list of the dataset. Each item of the list is a feature name of type str. Default: [].
- **dataset.labels** (list) - Label list of the dataset. Each item of the list is a label name of type str. Default: [].
- **epochs** (int) - epoch of the training process. Default: 1.
- **batch_size** (int) - Batch size of training data. Default: 1.
- **is_eval** (bool) - Whether execute evaluation after training. Default: False.
- **label_dp** (dict) - Configurations of the difference privacy algorithm. Default: {}.
- **label_dp.eps** (float) - eps of the difference privacy algorithm. Default: 1.0.
- **ckpt_path** (str) - Path to save checkpoints files保存训练网络checkpoint文件的路径. Default: "./checkpoints".

MindSpore Federated provides a demo project of [Vertical Federated Learning - Wide&Deep-based Recommendation Application](https://gitee.com/mindspore/federated/tree/r0.1/example/splitnn_criteo), which adopts the Wide&Deep model and the Criteo Dataset. Take the demo project as an example, the yaml configuration of the leader party of the vertical federated learning system is as follows:

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
