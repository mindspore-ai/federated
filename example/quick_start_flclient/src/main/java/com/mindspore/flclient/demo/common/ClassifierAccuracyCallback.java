/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

package com.mindspore.flclient.demo.common;

import com.mindspore.Model;
import com.mindspore.flclient.model.Callback;
import com.mindspore.flclient.model.CommonUtils;
import com.mindspore.flclient.model.Status;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Defining the Callback calculate classifier model.
 *
 * @since v1.0
 */
public class ClassifierAccuracyCallback extends Callback {
    private static final Logger LOGGER = Logger.getLogger(ClassifierAccuracyCallback.class.toString());
    private final int numOfClass;
    private final int batchSize;
    private final List<Integer> targetLabels;
    private float accuracy;

    private final String unsupervisedItemName = "DirectClassifierResult";

    private HashMap<String, float[]> classifierResult = new HashMap<>();

    /**
     * Defining a constructor of  ClassifierAccuracyCallback.
     */
    public ClassifierAccuracyCallback(Model model, int batchSize, int numOfClass, List<Integer> targetLabels) {
        super(model);
        this.batchSize = batchSize;
        this.numOfClass = numOfClass;
        this.targetLabels = targetLabels;
        float[] directClassifierResult = new float[numOfClass];
        classifierResult.put(unsupervisedItemName, directClassifierResult);
    }

    /**
     * Get eval accuracy.
     *
     * @return accuracy.
     */
    public float getAccuracy() {
        return accuracy;
    }

    /**
     * Get  ClassifierResult.
     *
     * @return ClassifierResult.
     */
    public HashMap<String, float[]> getClassifierResult() {
        return classifierResult;
    }


    @Override
    public Status stepBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status stepEnd() {
        Status status = calAccuracy();
        if (status != Status.SUCCESS) {
            return status;
        }

        status = calClassifierResult();
        if (status != Status.SUCCESS) {
            return status;
        }

        steps++;
        return Status.SUCCESS;
    }

    @Override
    public Status epochBegin() {
        return Status.SUCCESS;
    }

    @Override
    public Status epochEnd() {
        LOGGER.info("average accuracy:" + steps + ",acc is:" + accuracy / steps);
        accuracy = accuracy / steps;

        float[] directClassifierResult = classifierResult.get(unsupervisedItemName);
        for (int c = 0; c < numOfClass; c++) {
            directClassifierResult[c] /= steps * batchSize;
        }

        steps = 0;
        return Status.SUCCESS;
    }

    /***
     * Cal ClassifierResult for unsupervised train evaluate.
     * Now just return the mean of classifier result for saving communication data size while model update request
     * @return
     */
    private Status calClassifierResult() {
        Map<String, float[]> outputs = getOutputsBySize(batchSize * numOfClass);
        if (outputs.isEmpty()) {
            LOGGER.severe("Cannot find outputs tensor for calClassifierResult");
            return Status.FAILED;
        }

        Map.Entry<String, float[]> first = outputs.entrySet().iterator().next();
        float[] scores = first.getValue();

        if (scores.length != batchSize * numOfClass) {
            LOGGER.severe("Expect ClassifierResult length is:" + batchSize * numOfClass + ", but got " + scores.length);
            return Status.FAILED;
        }

        float[] directClassifierResult = classifierResult.get(unsupervisedItemName);
        for (int c = 0; c < numOfClass; c++) {
            if (steps == 0) {
                directClassifierResult[c] = 0.0f;
            }
            for (int b = 0; b < batchSize; b++) {
                directClassifierResult[c] += scores[numOfClass * b + c];
            }
        }

        LOGGER.info("DirectClassifierResult is:" + Arrays.toString(directClassifierResult));
        return Status.SUCCESS;
    }


    private Status calAccuracy() {
        if (targetLabels == null || targetLabels.isEmpty()) {
            LOGGER.severe("labels cannot be null");
            return Status.NULLPTR;
        }
        Map<String, float[]> outputs = getOutputsBySize(batchSize * numOfClass);
        if (outputs.isEmpty()) {
            LOGGER.severe("Cannot find outputs tensor for calAccuracy");
            return Status.FAILED;
        }
        Map.Entry<String, float[]> first = outputs.entrySet().iterator().next();
        float[] scores = first.getValue();
        int hitCounts = 0;
        for (int b = 0; b < batchSize; b++) {
            int predictIdx = CommonUtils.getMaxScoreIndex(scores, numOfClass * b, numOfClass * b + numOfClass);
            if (targetLabels.get(b + steps * batchSize) == predictIdx) {
                hitCounts += 1;
            }
        }
        accuracy += ((float) (hitCounts) / batchSize);
        LOGGER.info("steps:" + steps + ",acc is:" + (float) (hitCounts) / batchSize);
        return Status.SUCCESS;
    }
}
