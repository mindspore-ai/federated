/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

package com.mindspore.flclient.model;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.logging.Logger;

public class AUCCalculator {
    private static final Logger LOGGER = Logger.getLogger(AUCCalculator.class.toString());
    private List<Float> fps = new ArrayList<>();
    private List<Float> tps = new ArrayList<>();
    private List<Float> thresholds = new ArrayList<>();
    private List<Float> fpr = new ArrayList<>();
    private List<Float> tpr = new ArrayList<>();

    private Boolean isValidValue(Float val) {
        return !val.isInfinite() && !val.isNaN() && val >= 0.0 && val <= 1.0;
    }

    public Float getAuc(List<Float> label, List<Float> predict) {
        LOGGER.info("label is:" + label.toString());
        LOGGER.info("predict is:" + predict.toString());
        fps.clear();
        tps.clear();
        thresholds.clear();
        fpr.clear();
        tpr.clear();
        boolean ret = binaryClfCurve(label, predict);
        if (!ret) {
            LOGGER.severe("Do binaryClfCurve failed.");
            return 0.f;
        }
        ret = rocCurve();
        if (!ret) {
            LOGGER.severe("Do rocCurve failed.");
            return 0.f;
        }

        Float val = trapz(fpr, tpr);
        return val.isNaN() ? 0 : val;
    }

    private Boolean binaryClfCurve(List<Float> label, List<Float> predict) {
        if (label.size() != predict.size()) {
            LOGGER.severe("The input len of label is not same to predict.");
            return false;
        }
        for (int i = 0; i < label.size(); i++) {
            if (!isValidValue(label.get(i)) || !isValidValue(predict.get(i))) {
                LOGGER.severe("Get invalid value, idx " + i +
                        " label value:" + label.get(i) + " predict value:" + predict.get(i));
                return false;
            }
        }

        // start sort the pred && label by pred values
        // same to the sklearn using revert idx
        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < label.size(); i++) {
            idx.add(label.size() - 1 - i);
        }

        // get idx by desc of pred value
        idx.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                if (predict.get(o1) > predict.get(o2)) {
                    return -1;
                } else if (predict.get(o1).equals(predict.get(o2))) {
                    return 0;
                } else {
                    return 1;
                }
            }
        });

        // get sort pred and corresponding label values
        List<Float> sortLabel = new ArrayList<>();
        List<Float> sortPred = new ArrayList<>();
        for (int i = 0; i < idx.size(); i++) {
            int curIdx = idx.get(i);
            sortLabel.add(label.get(curIdx));
            sortPred.add(predict.get(curIdx));
        }

        // sortPred typically has many tied values. Here we extract
        // the indices associated with the distinct values.
        List<Integer> distinctValueIdx = new ArrayList<>();
        Float preVal = sortPred.get(0);
        for (int i = 1; i < sortPred.size(); i++) {
            Float curVal = sortPred.get(i);
            if (preVal.equals(curVal)) {
                continue;
            } else {
                preVal = curVal;
                distinctValueIdx.add(i - 1);
            }
        }

        distinctValueIdx.add(sortPred.size() - 1);
        Float sum = 0.0f;
        int pred_idx = 0;
        for (int i = 0; i < distinctValueIdx.size(); i++) {
            int cur_idx = distinctValueIdx.get(i);
            while (cur_idx >= pred_idx) {
                Float curVal = sortLabel.get(pred_idx);
                sum += curVal;
                pred_idx++;
            }
            tps.add(sum);
            fps.add(cur_idx + 1 - sum);
            thresholds.add(sortPred.get(cur_idx));
        }
        return true;
    }

    private List<Float> getDiff(List<Float> data) {
        List<Float> diff = new ArrayList<>();
        Float pred = data.get(0);
        for (int i = 1; i < data.size(); i++) {
            Float cur = data.get(i);
            diff.add(cur - pred);
            pred = cur;
        }
        return diff;
    }

    private List<Float> getDiffByLev(List<Float> data, int level) {
        int curDiffLev = 0;
        while (level > curDiffLev && data.size() > 1) {
            data = getDiff(data);
            curDiffLev++;
        }
        return data;
    }

    private Float trapz(List<Float> x, List<Float> y) {
        if (x.size() != y.size()) {
            throw new IllegalArgumentException("x.length != y.length");
        }
        if (y.size() == 0) {
            throw new IllegalArgumentException("y.length == 0");
        }
        Float value = 0.0f;
        Float x0 = x.get(0);
        Float y0 = y.get(0);
        for (int i = 1; i < y.size(); i++) {
            Float x1 = x.get(i);
            Float y1 = y.get(i);
            Float dx = x1 - x0;
            Float ym = y0 + y1;
            value += dx * ym;
            x0 = x1;
            y0 = y1;
        }
        return value / 2.0f;
    }

    private boolean rocCurve() {
        if (fps.size() > 2) {
            List<Float> fpsDiff2 = getDiffByLev(fps, 2);
            List<Float> tpsDiff2 = getDiffByLev(tps, 2);
            if (fpsDiff2.size() != tpsDiff2.size()) {
                LOGGER.severe("The size of fps_diff2 " + fpsDiff2.size() +
                        " is not same to tps_diff2 " + tpsDiff2.size());
                return false;
            }

            // Attempt to drop thresholds corresponding to points in between and
            // collinear with other points. These are always suboptimal and do not
            // appear on a plotted ROC curve (and thus do not affect the AUC).
            // Here getDiffByLev(_, 2) is used as a "second derivative" to tell if there
            // is a corner at the point. Both fps and tps must be tested to handle
            // thresholds with multiple data points (which are combined in
            // binaryClfCurve). This keeps all cases where the point should be kept,
            // but does not drop more complicated cases like fps = [1, 3, 7],
            // tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
            List<Integer> optimalIdxs = new ArrayList<>();
            optimalIdxs.add(0);
            for (int i = 0; i < fpsDiff2.size(); i++) {
                Float fpsDiffVal = fpsDiff2.get(i);
                Float tpsDiffVal = tpsDiff2.get(i);
                if (Math.abs(fpsDiffVal) + Math.abs(tpsDiffVal) > 0.0) {
                    optimalIdxs.add(i + 1);
                }
            }
            optimalIdxs.add(fpsDiff2.size() + 1);

            List<Float> optFps = new ArrayList<>();
            List<Float> optTps = new ArrayList<>();
            List<Float> optThresholds = new ArrayList<>();
            optFps.add(0f);
            optTps.add(0f);
            optThresholds.add(thresholds.get(0) + 1);

            for (int i = 0; i < optimalIdxs.size(); i++) {
                int idx = optimalIdxs.get(i);
                optFps.add(fps.get(idx));
                optTps.add(tps.get(idx));
                optThresholds.add(thresholds.get(idx));
            }
            fps = optFps;
            tps = optTps;
            thresholds = optThresholds;
        } else {
            fps.add(0, 0.f);
            tps.add(0, 0.f);
            Float addVal = thresholds.get(0) + 1;
            thresholds.add(0, addVal);
        }

        Float lastFps = fps.get(fps.size() - 1);
        Float lastTps = tps.get(tps.size() - 1);
        Boolean fprNanFlg = lastFps <= 0.0;
        Boolean tprNamFlg = lastTps <= 0.0;
        for (int i = 0; i < fps.size(); i++) {
            Float curFps = fps.get(i);
            Float curTps = tps.get(i);
            fpr.add(fprNanFlg ? Float.NaN : curFps / lastFps);
            tpr.add(tprNamFlg ? Float.NaN : curTps / lastTps);
        }
        return true;
    }
}
