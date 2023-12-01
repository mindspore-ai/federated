# MindSpore Federated Release Notes

[View English](./RELEASE.md)

## MindSpore Federated 0.2 Release Notes

MindSpore Federated是面向MindSpore开发者的开源联邦学习工具，支持机器学习的各参与方在不直接共享本地数据的前提下，共建AI模型。

### 主要特性及增强

* 隐私保护SignDS算法增强：基于局部差分隐私算法的均值估计统计任务，提供自适应调整重构模型梯度的能力。
* 联邦训练过程优化：去除每个iteration训练中的GetModel，替换使用GetResult进行同步确认，降低通讯量。
* 适配MindSpore 2.1.0。

### 贡献者

感谢以下人员做出的贡献：

Tang Cong, Zhang Zhugucheng, Ma Chenggui, Zhang Qi.

欢迎以任何形式对项目提供贡献！
