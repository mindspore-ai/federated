/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "distributed_cache/unsupervised_eval.h"

#include <algorithm>
#include <unordered_set>
#include "common/common.h"

namespace mindspore {
namespace fl {
namespace cache {

size_t UnsupervisedEval::cluster_argmax(const std::vector<float> &group_id) {
  return std::max_element(group_id.begin(), group_id.end()) - group_id.begin();
}

float UnsupervisedEval::calinski_harabasz_score(const std::vector<std::vector<float>> &X,
                                                const std::vector<size_t> &labels) {
  size_t n_samples = X.size();
  size_t label_size = X[0].size();
  std::unordered_set<size_t> nhash(labels.begin(), labels.end());
  size_t n_labels = nhash.size();
  if (n_labels <= 1) {
    return 0.0f;
  }

  float extra_disp = 0.0f;
  float intra_disp = 0.0f;
  std::vector<float> mean(label_size);
  for (size_t j = 0; j < label_size; j++) {
    float sum = 0.0f;
    for (size_t i = 0; i < n_samples; i++) {
      sum += X[i][j];
    }
    mean[j] = sum / n_samples;
  }

  for (size_t k = 0; k < label_size; k++) {
    std::vector<std::vector<float>> cluster_k;
    for (size_t i = 0; i < labels.size(); i++) {
      if (labels[i] == k) {
        cluster_k.emplace_back(X[i]);
      }
    }
    size_t cluster_k_size = cluster_k.size();
    if (cluster_k_size == 0) {
      continue;
    }
    std::vector<float> mean_k(label_size);
    for (size_t j = 0; j < label_size; j++) {
      float sum = 0.0f;
      for (size_t i = 0; i < cluster_k_size; i++) {
        sum += cluster_k[i][j];
      }
      mean_k[j] = sum / cluster_k.size();
    }

    float sum = 0.0f;
    for (size_t i = 0; i < mean_k.size(); i++) {
      sum += std::pow(mean_k[i] - mean[i], 2);
    }

    extra_disp += cluster_k_size * sum;
    sum = 0.0f;
    for (size_t i = 0; i < cluster_k_size; i++) {
      for (size_t j = 0; j < label_size; j++) {
        sum += std::pow(cluster_k[i][j] - mean_k[j], 2);
      }
    }
    intra_disp += sum;
  }

  if (intra_disp == 0.0f) {
    return 1.0f;
  } else {
    return extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0));
  }
}
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
