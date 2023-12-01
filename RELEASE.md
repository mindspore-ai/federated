# MindSpore Federated Release Notes

[查看中文](./RELEASE_CN.md)

## MindSpore Federated 0.2 Release Notes

MindSpore Federated is an open source federated learning tool for MindSpor, supports the various participants of machine learning to build AI models together without directly sharing local data.
To improve the usability of the federated framework and user development efficiency, related code has been separated from the MindSpore framework and Federated is now an independent repository.

### Major Features and Improvements

* Privacy preserving SignDS algorithm improvement: The statistical task of mean estimation based on the local differential privacy algorithm provides the capability of adaptively adjusting the gradient of the reconstructed model.
* Federated training process optimization: Remove GetModel in each iteration training and use GetResult for synchronization confirmation, reducing communication traffic.
* Adapts to MindSpore 2.1.0.

### Contributors

Thanks goes to these wonderful people:

Tang Cong, Zhang Zhugucheng, Ma Chenggui, Zhang Qi.

Contributions of any kind are welcome!
