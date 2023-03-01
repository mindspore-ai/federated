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
#include <unordered_map>
#include <limits>
#include "common/common.h"

namespace mindspore {
namespace fl {
namespace cache {

size_t UnsupervisedEval::clusterArgmax(const std::vector<float> &group_id) {
  return std::max_element(group_id.begin(), group_id.end()) - group_id.begin();
}

float UnsupervisedEval::clusterEvaluate(const std::vector<std::vector<float>> &group_ids,
                                        const std::vector<size_t> &labels, const std::string &eval_type) {
  float score = 0.0f;
  if (eval_type == kSilhouetteScoreType) {
    score = silhouetteScore(group_ids, labels);
  } else if (eval_type == kCalinskiHarabaszScoreType) {
    score = calinskiHarabaszScore(group_ids, labels);
  } else if (eval_type == kDaviesBouldinScoreType) {
    score = daviesBouldinScore(group_ids, labels);
  } else {
    MS_LOG(EXCEPTION) << "Eval type:" << eval_type << " is not valid.";
  }
  return score;
}

float UnsupervisedEval::calinskiHarabaszScore(const std::vector<std::vector<float>> &group_ids,
                                              const std::vector<size_t> &labels) {
  if (group_ids.empty() || labels.empty()) {
    return false;
  }
  size_t n_samples = group_ids.size();
  size_t label_size = group_ids[0].size();
  std::unordered_set<size_t> nhash_labels(labels.begin(), labels.end());
  size_t n_labels = nhash_labels.size();
  if (n_labels < 2 || n_labels > n_samples - 1) {
    MS_LOG(WARNING) << "Number of n_labels: " << n_labels << " is invalid, valid values are 2 to n_samples - 1.";
    return 0.0f;
  }

  float extra_disp = 0.0f;
  float intra_disp = 0.0f;
  std::vector<float> mean(label_size);
  for (size_t j = 0; j < label_size; j++) {
    float sum = 0.0f;
    for (size_t i = 0; i < n_samples; i++) {
      sum += group_ids[i][j];
    }
    mean[j] = sum / n_samples;
  }

  for (size_t k = 0; k < label_size; k++) {
    std::vector<std::vector<float>> cluster_k;
    for (size_t i = 0; i < labels.size(); i++) {
      if (labels[i] == k) {
        cluster_k.emplace_back(group_ids[i]);
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

std::vector<std::vector<float>> UnsupervisedEval::euclideanDistanceMatrix(
  const std::vector<std::vector<float>> &group_ids) {
  size_t n_samples = group_ids.size();
  std::vector<std::vector<float>> distance_matrix(n_samples);
  for (size_t i = 0; i < n_samples; i++) {
    distance_matrix[i].resize(i);
  }

  for (size_t i = 0; i < n_samples; ++i) {
    for (size_t j = i + 1; j < n_samples; ++j) {
      float distance = 0.0f;
      for (size_t k = 0; k < group_ids[i].size(); ++k) {
        float diff = group_ids[i][k] - group_ids[j][k];
        distance += diff * diff;
      }
      distance_matrix[j][i] = std::sqrt(distance);
    }
  }
  return distance_matrix;
}

float UnsupervisedEval::silhouetteScore(const std::vector<std::vector<float>> &group_ids,
                                        const std::vector<size_t> &labels) {
  if (group_ids.empty() || labels.empty()) {
    return false;
  }
  size_t n_samples = group_ids.size();
  std::vector<float> s_i(n_samples);

  std::unordered_set<size_t> nhash_labels(labels.begin(), labels.end());
  size_t n_labels = nhash_labels.size();
  if (n_labels < 2 || n_labels > n_samples - 1) {
    MS_LOG(WARNING) << "Number of n_labels: " << n_labels << " is invalid, valid values are 2 to n_samples - 1.";
    return 0.0f;
  }
  auto distance_matrix = euclideanDistanceMatrix(group_ids);
  for (size_t i = 0; i < n_samples; ++i) {
    std::vector<float> a_distances;
    float a_i = 0.0;
    float b_i = std::numeric_limits<float>::max();
    size_t label_i = labels[i];
    std::unordered_map<size_t, std::vector<float>> b_i_map;
    for (size_t j = 0; j < n_samples; ++j) {
      if (i == j) {
        continue;
      }
      size_t label_j = labels[j];
      float distance = 0.0f;
      if (i > j) {
        distance = distance_matrix[i][j];
      } else {
        distance = distance_matrix[j][i];
      }
      if (label_j == label_i) {
        a_distances.push_back(distance);
      } else {
        b_i_map[label_j].push_back(distance);
      }
    }
    if (a_distances.size() > 0) {
      a_i = std::accumulate(a_distances.begin(), a_distances.end(), 0.0) / a_distances.size();
    }
    for (auto &item : b_i_map) {
      auto &b_i_distances = item.second;
      float b_i_distance = std::accumulate(b_i_distances.begin(), b_i_distances.end(), 0.0) / b_i_distances.size();
      b_i = std::min(b_i, b_i_distance);
    }
    if (a_i == 0) {
      s_i[i] = 0;
    } else {
      s_i[i] = (b_i - a_i) / std::max(a_i, b_i);
    }
  }
  return std::accumulate(s_i.begin(), s_i.end(), 0.0) / n_samples;
}

float UnsupervisedEval::euclideanDistance(const std::vector<float> &id1, const std::vector<float> &id2) {
  if (id1.size() != id2.size()) {
    MS_LOG(WARNING) << "Group-IDs has different data dimensions.";
    return 0.0f;
  }
  float sum = 0.0f;
  for (size_t i = 0; i < id1.size(); i++) {
    sum += (id1[i] - id2[i]) * (id1[i] - id2[i]);
  }
  return sqrt(sum);
}

std::vector<std::vector<float>> UnsupervisedEval::getCentroid(const std::vector<std::vector<float>> &group_ids,
                                                              const std::vector<size_t> &labels) {
  size_t n_samples = group_ids.size();
  std::unordered_set<size_t> nhash_labels(labels.begin(), labels.end());
  size_t n_labels = nhash_labels.size();
  size_t n_feature = group_ids[0].size();
  std::vector<std::vector<float>> centroid(n_labels);
  for (size_t i = 0; i < n_labels; i++) {
    centroid[i].resize(n_feature);
  }
  for (size_t i = 0; i < n_samples; i++) {
    size_t labeli = labels[i];
    for (size_t j = 0; j < n_feature; j++) {
      centroid[labeli][j] += group_ids[i][j];
    }
  }
  for (size_t i = 0; i < n_labels; i++) {
    size_t n_samples_i = std::count(labels.begin(), labels.end(), i);
    for (size_t j = 0; j < n_feature; j++) {
      centroid[i][j] /= n_samples_i;
    }
  }
  return centroid;
}

std::vector<float> UnsupervisedEval::getIntra(const std::vector<std::vector<float>> &group_ids,
                                              const std::vector<std::vector<float>> &centroid,
                                              const std::vector<size_t> &labels) {
  size_t n_samples = group_ids.size();
  size_t n_labels = centroid.size();
  std::vector<float> intra(n_labels);
  for (size_t i = 0; i < n_samples; i++) {
    size_t labeli = labels[i];
    intra[labeli] += euclideanDistance(group_ids[i], centroid[labeli]);
  }
  for (size_t i = 0; i < n_labels; i++) {
    size_t n_samples_i = std::count(labels.begin(), labels.end(), i);
    intra[i] /= n_samples_i;
  }
  return intra;
}

std::vector<std::vector<float>> UnsupervisedEval::getOutra(const std::vector<std::vector<float>> &centroid) {
  size_t n_labels = centroid.size();
  std::vector<std::vector<float>> outra(n_labels - 1);
  for (size_t i = 0; i < n_labels - 1; i++) {
    outra[i].resize(n_labels - 1 - i);
  }
  for (size_t i = 0; i < n_labels - 1; i++) {
    for (size_t j = 0; j < n_labels - 1 - i; j++) {
      outra[i][j] = euclideanDistance(centroid[i], centroid[i + 1 + j]);
    }
  }
  return outra;
}

std::vector<std::vector<float>> UnsupervisedEval::getCombined(const std::vector<float> intra,
                                                              const std::vector<std::vector<float>> &outra) {
  size_t n_labels = intra.size();
  std::vector<std::vector<float>> combined(n_labels - 1);
  for (size_t i = 0; i < n_labels - 1; i++) {
    combined[i].resize(n_labels - 1 - i);
  }
  for (size_t i = 0; i < n_labels - 1; i++) {
    for (size_t j = 0; j < n_labels - 1 - i; j++) {
      combined[i][j] = (intra[i] + intra[j + i + 1]) / outra[i][j];
    }
  }
  return combined;
}

std::vector<size_t> UnsupervisedEval::findPosition(const size_t i, const size_t j) {
  std::vector<size_t> pos(2);
  if (i < j) {
    pos[0] = i;
    pos[1] = j - 1 - i;
  } else {
    pos[0] = j;
    pos[1] = i - 1 - j;
  }
  return pos;
}

float UnsupervisedEval::daviesBouldinScore(const std::vector<std::vector<float>> &group_ids,
                                           const std::vector<size_t> &labels) {
  size_t n_samples = group_ids.size();
  std::unordered_set<size_t> nhash_labels(labels.begin(), labels.end());
  size_t n_labels = nhash_labels.size();
  if (n_labels < 2 || n_labels > n_samples - 1) {
    MS_LOG(WARNING) << "Number of n_labels: " << n_labels << " is invalid, valid values are 2 to n_samples - 1.";
    return 0.0f;
  }
  auto centroid = getCentroid(group_ids, labels);
  auto intra = getIntra(group_ids, centroid, labels);
  auto outra = getOutra(centroid);
  auto combined = getCombined(intra, outra);
  std::vector<float> score(n_labels);
  for (size_t i = 0; i < n_labels; i++) {
    size_t row = (i < 1) ? i : (i - 1);
    float max = combined[row][0];
    for (size_t j = 0; j < n_labels; j++) {
      if (i == j) continue;
      auto pos = findPosition(i, j);
      if (combined[pos[0]][pos[1]] > max) {
        max = combined[pos[0]][pos[1]];
      }
    }
    score[i] = max;
  }
  return std::accumulate(score.begin(), score.end(), 0.0f) / n_labels;
}
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
