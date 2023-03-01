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
#ifndef MINDSPORE_FL_CACHE_UNSUPERVISED_EVAL_H
#define MINDSPORE_FL_CACHE_UNSUPERVISED_EVAL_H
#include <string>
#include <vector>

namespace mindspore {
namespace fl {
namespace cache {
class UnsupervisedEval {
 public:
  static UnsupervisedEval &Instance() {
    static UnsupervisedEval instance;
    return instance;
  }

  static size_t clusterArgmax(const std::vector<float> &group_id);
  static float calinskiHarabaszScore(const std::vector<std::vector<float>> &group_ids,
                                     const std::vector<size_t> &labels);

  std::vector<std::vector<float>> euclideanDistanceMatrix(const std::vector<std::vector<float>> &group_ids);
  float silhouetteScore(const std::vector<std::vector<float>> &data, const std::vector<size_t> &labels);
  float clusterEvaluate(const std::vector<std::vector<float>> &group_ids, const std::vector<size_t> &labels,
                        const std::string &eval_type);

  static float daviesBouldinScore(const std::vector<std::vector<float>> &group_ids, const std::vector<size_t> &labels);
  static std::vector<std::vector<float>> getCentroid(const std::vector<std::vector<float>> &group_ids,
                                                     const std::vector<size_t> &labels);
  static std::vector<float> getIntra(const std::vector<std::vector<float>> &group_ids,
                                     const std::vector<std::vector<float>> &centroid,
                                     const std::vector<size_t> &labels);
  static std::vector<std::vector<float>> getOutra(const std::vector<std::vector<float>> &centroid);
  static std::vector<std::vector<float>> getCombined(const std::vector<float> intra,
                                                     const std::vector<std::vector<float>> &outra);
  static float euclideanDistance(const std::vector<float> &id1, const std::vector<float> &id2);
  static std::vector<size_t> findPosition(const size_t i, const size_t j);
  static std::vector<float> dbMatrix(const std::vector<std::vector<float>> &Rij);
};
}  // namespace cache
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_FL_CACHE_UNSUPERVISED_EVAL_H
