/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

package com.mindspore.flclient;

import com.google.flatbuffers.FlatBufferBuilder;

import com.mindspore.flclient.common.FLLoggerGenerater;
import com.mindspore.flclient.model.Client;
import mindspore.fl.schema.FeatureMap;

import java.security.SecureRandom;
import java.util.*;
import java.util.logging.Logger;

import static java.lang.Math.log;

/**
 * Defines encryption and decryption methods.
 *
 * @since 2021-06-30
 */
public class SecureProtocol {
    private static final Logger LOGGER = FLLoggerGenerater.getModelLogger(SecureProtocol.class.toString());
    private static double deltaError = 1e-6d;
    private static float laplaceEpsUpper = 500000f;

    private static Map<String, float[]> modelMap;

    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private int iteration;
    private CipherClient cipherClient;
    private FLClientStatus status;
    private float[] featureMask = new float[0];
    private double dpEps;
    private double dpDelta;
    private double dpNormClip;
    private ArrayList<String> updateFeatureName = new ArrayList<String>();
    private int retCode;
    private float signK;
    private float signEps;
    private float signThrRatio;
    private int signDimOut;


    /**
     * Obtain current status code in client.
     *
     * @return current status code in client.
     */
    public FLClientStatus getStatus() {
        return status;
    }

    /**
     * Obtain retCode returned by server.
     *
     * @return the retCode returned by server.
     */
    public int getRetCode() {
        return retCode;
    }

    /**
     * Setting parameters for pairwise masking.
     *
     * @param iter         current iteration of federated learning task.
     * @param minSecretNum minimum number of secret fragments required to reconstruct a secret
     * @param prime        teh big prime number used to split secrets into pieces
     * @param featureSize  the total feature size in model
     */
    public void setPWParameter(int iter, int minSecretNum, byte[] prime, int featureSize) {
        if (prime == null || prime.length == 0) {
            LOGGER.severe("[PairWiseMask] the input argument <prime> is null, please check!");
            throw new IllegalArgumentException();
        }
        this.iteration = iter;
        this.cipherClient = new CipherClient(iteration, minSecretNum, prime, featureSize);
    }

    /**
     * Setting parameters for differential privacy.
     *
     * @param iter      current iteration of federated learning task.
     * @param diffEps   privacy budget eps of DP mechanism.
     * @param diffDelta privacy budget delta of DP mechanism.
     * @param diffNorm  normClip factor of DP mechanism.
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus setDPParameter(int iter, double diffEps, double diffDelta, double diffNorm) {
        this.iteration = iter;
        this.dpEps = diffEps;
        this.dpDelta = diffDelta;
        this.dpNormClip = diffNorm;
        return FLClientStatus.SUCCESS;
    }

    /**
     * Setting parameters for dimension select.
     *
     * @param signK
     * @param signEps
     * @param signThrRatio
     * @param signDimOut
     * @return
     */
    public FLClientStatus setDSParameter(float signK, float signEps, float signThrRatio,
                                         int signDimOut) {
        this.signK = signK;
        this.signEps = signEps;
        this.signThrRatio = signThrRatio;
        this.signDimOut = signDimOut;
        return FLClientStatus.SUCCESS;
    }

    /**
     * Obtain the feature names that needed to be encrypted.
     *
     * @return the feature names that needed to be encrypted.
     */
    public ArrayList<String> getUpdateFeatureName() {
        return updateFeatureName;
    }

    /**
     * Set the parameter updateFeatureName.
     *
     * @param updateFeatureName the feature names that needed to be encrypted.
     */
    public void setUpdateFeatureName(ArrayList<String> updateFeatureName) {
        this.updateFeatureName = updateFeatureName;
    }

    /**
     * Obtain the returned timestamp for next request from server.
     *
     * @return the timestamp for next request.
     */
    public String getNextRequestTime() {
        return cipherClient.getNextRequestTime();
    }


    /**
     * Get the DpNormClip
     *
     * @return dpNormClip
     */
    public double getDpNormClip() {
        return dpNormClip;
    }

    /**
     * Generate pairwise mask and individual mask.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus pwCreateMask() {
        LOGGER.info(String.format("[PairWiseMask] ==============request flID: %s ==============",
                localFLParameter.getFlID()));
        // round 0
        if (localFLParameter.isStopJobFlag()) {
            LOGGER.info("the stopJObFlag is set to true, the job will be stop");
            return status;
        }
        status = cipherClient.exchangeKeys();
        retCode = cipherClient.getRetCode();
        LOGGER.info(String.format("[PairWiseMask] ============= RequestExchangeKeys+GetExchangeKeys response: %s ",
                "============", status));
        if (status != FLClientStatus.SUCCESS) {
            return status;
        }
        // round 1
        if (localFLParameter.isStopJobFlag()) {
            LOGGER.info("the stopJObFlag is set to true, the job will be stop");
            return status;
        }
        status = cipherClient.shareSecrets();
        retCode = cipherClient.getRetCode();
        LOGGER.info(String.format("[Encrypt] =============RequestShareSecrets+GetShareSecrets response: %s ",
                "=============", status));
        if (status != FLClientStatus.SUCCESS) {
            return status;
        }
        // round2
        if (localFLParameter.isStopJobFlag()) {
            LOGGER.info("the stopJObFlag is set to true, the job will be stop");
            return status;
        }
        featureMask = cipherClient.doubleMaskingWeight();
        if (featureMask == null || featureMask.length <= 0) {
            LOGGER.severe("[Encrypt] the returned featureMask from cipherClient.doubleMaskingWeight" +
                    " is null, please check!");
            return FLClientStatus.FAILED;
        }
        retCode = cipherClient.getRetCode();
        LOGGER.info("[Encrypt] =============Create double feature mask: SUCCESS=============");
        return status;
    }

    /**
     * Add the pairwise mask and individual mask to one weight.
     *
     * @param trainDataSize the record num of train data
     * @param feature       the feature that need to do pwMask
     * @param maskIndex     the start pos of featureMask
     * @return encrypt result
     */
    public float[] pwMaskWeight(int trainDataSize, float[] feature, int maskIndex) {
        Map<String, List<Float>> featureMaps = new HashMap<>();
        if (featureMask == null || featureMask.length == 0) {
            throw new RuntimeException("[pwMaskWeight] feature mask is null, please check");
        }

        if (featureMask.length < maskIndex + feature.length) {
            throw new RuntimeException("[pwMaskWeight] the data length is out of range for array featureMask, " +
                    "featureMask length:" + featureMask.length + " data length:" + feature.length);
        }

        LOGGER.info(String.format("[pwMaskWeight] feature mask size: %s", featureMask.length));
        float[] maskedData = new float[feature.length];
        LOGGER.info(String.format("[pwMaskWeight] feature  size: %s", feature.length));
        for (int j = 0; j < feature.length; j++) {
            maskedData[j] = feature[j] * trainDataSize + featureMask[maskIndex + j];
        }
        return maskedData;
    }


    /**
     * Reconstruct the secrets used for unmasking model weights.
     *
     * @return current status code in client.
     */
    public FLClientStatus pwUnmasking() {
        status = cipherClient.reconstructSecrets();   // round3
        retCode = cipherClient.getRetCode();
        LOGGER.info(String.format("[Encrypt] =============GetClientList+SendReconstructSecret: %s =============",
                status));
        return status;
    }

    private static float calculateErf(double erfInput) {
        double result = 0d;
        int segmentNum = 10000;
        double deltaX = erfInput / segmentNum;
        result += 1;
        for (int i = 1; i < segmentNum; i++) {
            result += 2 * Math.exp(-Math.pow(deltaX * i, 2));
        }
        result += Math.exp(-Math.pow(deltaX * segmentNum, 2));
        return (float) (result * deltaX / Math.pow(Math.PI, 0.5));
    }

    private static double calculatePhi(double phiInput) {
        return 0.5 * (1.0 + calculateErf((phiInput / Math.sqrt(2.0))));
    }

    private static double calculateBPositive(double eps, double calInput) {
        return calculatePhi(Math.sqrt(eps * calInput)) -
                Math.exp(eps) * calculatePhi(-Math.sqrt(eps * (calInput + 2.0)));
    }

    private static double calculateBNegative(double eps, double calInput) {
        return calculatePhi(-Math.sqrt(eps * calInput)) -
                Math.exp(eps) * calculatePhi(-Math.sqrt(eps * (calInput + 2.0)));
    }

    private static double calculateSPositive(double eps, double targetDelta, double initSInf, double initSSup) {
        double deltaSup = calculateBPositive(eps, initSSup);
        double sInf = initSInf;
        double sSup = initSSup;
        while (deltaSup <= targetDelta) {
            sInf = sSup;
            sSup = 2 * sInf;
            deltaSup = calculateBPositive(eps, sSup);
        }
        double sMid = sInf + (sSup - sInf) / 2.0;
        int iterMax = 1000;
        int iters = 0;
        while (true) {
            double bPositive = calculateBPositive(eps, sMid);
            if (bPositive <= targetDelta) {
                if (targetDelta - bPositive <= deltaError) {
                    break;
                } else {
                    sInf = sMid;
                }
            } else {
                sSup = sMid;
            }
            sMid = sInf + (sSup - sInf) / 2.0;
            iters += 1;
            if (iters > iterMax) {
                break;
            }
        }
        return sMid;
    }

    private static double calculateSNegative(double eps, double targetDelta, double initSInf, double initSSup) {
        double deltaSup = calculateBNegative(eps, initSSup);
        double sInf = initSInf;
        double sSup = initSSup;
        while (deltaSup > targetDelta) {
            sInf = sSup;
            sSup = 2 * sInf;
            deltaSup = calculateBNegative(eps, sSup);
        }

        double sMid = sInf + (sSup - sInf) / 2.0;
        int iterMax = 1000;
        int iters = 0;
        while (true) {
            double bNegative = calculateBNegative(eps, sMid);
            if (bNegative <= targetDelta) {
                if (targetDelta - bNegative <= deltaError) {
                    break;
                } else {
                    sSup = sMid;
                }
            } else {
                sInf = sMid;
            }
            sMid = sInf + (sSup - sInf) / 2.0;
            iters += 1;
            if (iters > iterMax) {
                break;
            }
        }
        return sMid;
    }

    public double calculateSigma() {
        double deltaZero = calculateBPositive(dpEps, 0);
        double alpha = 1d;
        if (dpDelta > deltaZero) {
            double sPositive = calculateSPositive(dpEps, dpDelta, 0, 1);
            alpha = Math.sqrt(1.0 + sPositive / 2.0) - Math.sqrt(sPositive / 2.0);
        } else if (dpDelta < deltaZero) {
            double sNegative = calculateSNegative(dpEps, dpDelta, 0, 1);
            alpha = Math.sqrt(1.0 + sNegative / 2.0) + Math.sqrt(sNegative / 2.0);
        } else {
            LOGGER.info("[Encrypt] targetDelta = deltaZero");
        }
        return alpha * dpNormClip / Math.sqrt(2.0 * dpEps);
    }

    /**
     * The number of combinations of n things taken k.
     *
     * @param n Number of things.
     * @param k Number of elements taken.
     * @return the total number of "n choose k" combinations.
     */
    private static double comb(double n, double k) {
        boolean cond = (k <= n) && (n >= 0) && (k >= 0);
        double m = n + 1;
        if (!cond) {
            return 0;
        } else {
            double nTerm = Math.min(k, n - k);
            double res = 1;
            for (int i = 1; i <= nTerm; i++) {
                res *= (m - i);
                res /= i;
            }
            return res;
        }
    }

    /**
     * Calculate the number of possible combinations of output set given the number of topk dimensions.
     * c(k, v) * c(d-k, h-v)
     *
     * @param numInter  the number of dimensions from topk set.
     * @param topkDim   the size of top-k set.
     * @param inputDim  total number of dimensions in the model.
     * @param outputDim the number of dimensions selected for constructing sparse local updates.
     * @return the number of possible combinations of output set.
     */
    private static double countCombs(int numInter, int topkDim, int inputDim, int outputDim) {
        return comb(topkDim, numInter) * comb(inputDim - topkDim, outputDim - numInter);
    }

    /**
     * Calculate the probability mass function of the number of topk dimensions in the output set.
     * v is the number of dimensions from topk set.
     *
     * @param thr       threshold of the number of topk dimensions in the output set.
     * @param topkDim   the size of top-k set.
     * @param inputDim  total number of dimensions in the model.
     * @param outputDim the number of dimensions selected for constructing sparse local updates.
     * @param eps       the privacy budget of SignDS alg.
     * @return the probability mass function.
     */
    private static List<Double> calcPmf(int thr, int topkDim, int inputDim, int outputDim, float eps) {
        List<Double> pmf = new ArrayList<>();
        double newPmf;
        for (int v = 0; v <= outputDim; v++) {
            if (v < thr) {
                newPmf = countCombs(v, topkDim, inputDim, outputDim);
            } else {
                newPmf = countCombs(v, topkDim, inputDim, outputDim) * Math.exp(eps);
            }
            pmf.add(newPmf);
        }
        double pmfSum = 0;
        for (int i = 0; i < pmf.size(); i++) {
            pmfSum += pmf.get(i);
        }
        if (pmfSum == 0) {
            LOGGER.severe("[SignDS] probability mass function is 0, please check");
            return new ArrayList<>();
        }
        for (int i = 0; i < pmf.size(); i++) {
            pmf.set(i, pmf.get(i) / pmfSum);
        }
        return pmf;
    }

    /**
     * Calculate the expected number of topk dimensions in the output set given outputDim.
     * The size of pmf is also outputDim.
     *
     * @param pmf probability mass function
     * @return the expectation of the topk dimensions in the output set.
     */
    private static double calcExpectation(List<Double> pmf) {
        double sumExpectation = 0;
        for (int i = 0; i < pmf.size(); i++) {
            sumExpectation += (i * pmf.get(i));
        }
        return sumExpectation;
    }

    /**
     * Calculate the optimum threshold for the number of topk dimension in the output set.
     * The optimum threshold is an integer among [1, outputDim], which has the largest
     * expectation value.
     *
     * @param topkDim   the size of top-k set.
     * @param inputDim  total number of dimensions in the model.
     * @param outputDim the number of dimensions selected for constructing sparse local updates.
     * @param eps       the privacy budget of SignDS alg.
     * @return the optimum threshold.
     */
    private static int calcOptThr(int topkDim, int inputDim, int outputDim, float eps) {
        double optExpect = 0;
        double optT = 0;
        for (int t = 1; t <= outputDim; t++) {
            double newExpect = calcExpectation(calcPmf(t, topkDim, inputDim, outputDim, eps));
            if (newExpect > optExpect) {
                optExpect = newExpect;
                optT = t;
            } else {
                break;
            }
        }
        return (int) Math.max(optT, 1);
    }

    /**
     * Tool function for finding the optimum output dimension.
     * The main idea is to iteratively search for the largest output dimension while
     * ensuring the expected ratio of topk dimensions in the output set larger than
     * the target ratio.
     *
     * @param thrInterRatio threshold of the expected ratio of topk dimensions
     * @param topkDim       the size of top-k set.
     * @param inputDim      total number of dimensions in the model.
     * @param eps           the privacy budget of SignDS alg.
     * @return the optimum output dimension.
     */
    public static int findOptOutputDim(float thrInterRatio, int topkDim, int inputDim, float eps) {
        int outputDim = 1;
        while (true) {
            int thr = calcOptThr(topkDim, inputDim, outputDim, eps);
            double expectedRatio = calcExpectation(calcPmf(thr, topkDim, inputDim, outputDim, eps)) / outputDim;
            if (expectedRatio < thrInterRatio || Double.isNaN(expectedRatio)) {
                break;
            } else {
                outputDim += 1;
            }
        }
        return Math.max(1, (outputDim - 1));
    }

    /**
     * Determine the number of dimensions to be sampled from the topk dimension set via
     * inverse sampling.
     * The main steps of the trick of inverse sampling include:
     * 1. Sample a random probability from the uniform distribution U(0, 1).
     * 2. Calculate the cumulative distribution of numInter, namely the number of
     * topk dimensions in the output set.
     * 3. Compare the cumulative distribution with the random probability and determine
     * the value of numInter.
     *
     * @param thrDim      threshold of the number of topk dimensions in the output set.
     * @param denominator calculate denominator given the threshold.
     * @param topkDim     the size of top-k set.
     * @param inputDim    total number of dimensions in the model.
     * @param outputDim   the number of dimensions selected for constructing sparse local updates.
     * @param eps         the privacy budget of SignDS alg.
     * @return the number of dimensions to be sampled from the top-k dimension set.
     */
    private static int countInters(int thrDim, double denominator, int topkDim, int inputDim, int outputDim, float eps) {
        SecureRandom secureRandom = new SecureRandom();
        double randomProb = secureRandom.nextDouble();
        int numInter = 0;
        double prob = countCombs(numInter, topkDim, inputDim, outputDim) / denominator;
        while (prob < randomProb) {
            numInter += 1;
            if (numInter < thrDim) {
                prob += countCombs(numInter, topkDim, inputDim, outputDim) / denominator;
            } else {
                prob += Math.exp(eps) * countCombs(numInter, topkDim, inputDim, outputDim) / denominator;
            }
        }
        return numInter;
    }

    /**
     * elect num indexes from inputList, and put them into outputList.
     *
     * @param secureRandom cryptographically strong random number generator.
     * @param inputList    select index from inputList.
     * @param inStartPos   the start pos of inputList
     * @param inRang       the select range of inputList
     * @param selectNums   the number of select indexes.
     * @param outputList   put random index into outputList.
     * @param outStartPos  the start pos of outputList
     */
    private static void randomSelect(SecureRandom secureRandom, int[] inputList, int inStartPos, int inRang, int selectNums,
                                     int[] outputList, int outStartPos) {
        if (selectNums < 0) {
            LOGGER.severe("[SignDS] The number to be selected is set incorrectly!");
            return;
        }
        if (inputList.length < inStartPos + selectNums ||
                inputList.length < inStartPos + inRang || inRang < selectNums) {
            LOGGER.severe("[SignDS] The size of inputList is too small! inputList size:" +
                    inputList.length + " inStartPos:" + inStartPos + " inRang:" + inRang + " selectNums:" + selectNums);
            return;
        }

        for (int i = inRang; i > inRang - selectNums; i--) {
            int randomIndex = secureRandom.nextInt(i);
            int randomSelectTopkIndex = inputList[randomIndex + inStartPos];
            inputList[randomIndex + inStartPos] = inputList[i - 1 + inStartPos];
            inputList[i - 1 + inStartPos] = randomSelectTopkIndex;
            outputList[outStartPos + inRang - i] = randomSelectTopkIndex;
        }
    }


    private interface CompareOp {
        boolean operation(float l, float r);
    }

    private void merge(float[] data, int[] origIdx, int[] dstIdx,
                       int lPos, int lLen, int rPos, int rLen, CompareOp op) {
        int lIdx = 0;
        int rIdx = 0;

        while (lIdx < lLen && rIdx < rLen) {
            if (op.operation(data[origIdx[lPos + lIdx]], data[origIdx[rPos + rIdx]])) {
                dstIdx[lPos + lIdx + rIdx] = origIdx[lPos + lIdx];
                lIdx++;
            } else {
                dstIdx[lPos + lIdx + rIdx] = origIdx[rPos + rIdx];
                rIdx++;
            }
        }
        while (lIdx < lLen) {
            dstIdx[lPos + lIdx + rIdx] = origIdx[lPos + lIdx];
            lIdx++;
        }
        while (rIdx < rLen) {
            dstIdx[lPos + lIdx + rIdx] = origIdx[rPos + rIdx];
            rIdx++;
        }
    }

    private int[] mergeShort(float[] data, boolean sign) {
        CompareOp cmpAsc = (float l, float r) -> {
            return l < r;
        };
        CompareOp cmpDesc = (float l, float r) -> {
            return l > r;
        };
        CompareOp cmpOp = sign ? cmpDesc : cmpAsc;

        int dataSize = data.length;
        int dstIdx[] = new int[dataSize];
        int cacheIdx[] = new int[dataSize];
        for (int i = 0; i < dataSize; i++) {
            dstIdx[i] = i;
        }
        int sorted_len = 1;
        while (sorted_len < dataSize) {
            int i = 0;
            while (i < dataSize) {
                int lPos = i;
                int lLen = sorted_len;
                if (dataSize - i <= sorted_len) {
                    break;
                }
                i += sorted_len;
                int rPos = i;
                int rLen = sorted_len;
                if (dataSize - i <= sorted_len) {
                    rLen = dataSize - i;
                }
                merge(data, dstIdx, cacheIdx, lPos, lLen, rPos, rLen, cmpOp);
                i += rLen;
            }
            int[] tmp = dstIdx;
            dstIdx = cacheIdx;
            cacheIdx = tmp;
            sorted_len += sorted_len;
        }

        return dstIdx;
    }

    /**
     * Privacy preserving for input Boolean values.
     *
     * @param inputBool         Input Boolean values.
     * @param randomResponseEps Privacy Budget.
     * @return Boolean value after perturbation.
     */
    private static boolean randomResponse(boolean inputBool, double randomResponseEps) {
        SecureRandom secureRandom = Common.getSecureRandom();
        double probability = secureRandom.nextDouble();
        double threshold = Math.exp(randomResponseEps) / (1.0f + Math.exp(randomResponseEps));
        if (probability < threshold) {
            return inputBool;
        } else {
            return !inputBool;
        }
    }

    /**
     * Magnitude Random Response alg.
     *
     * @param originData        origin update data.
     * @param sortedIdx         sorted index.
     * @param rEst              global lr which downloaded from server.
     * @param randomResponseEps privacy budget for magRR.
     * @return Boolean value after perturbation.
     */
    private static boolean magRR(float[] originData, int[] sortedIdx, int topkDim, float rEst, double randomResponseEps, int rangeReached) {
        float avg = 0;
        final float MULTIPLE = 2.0f;
        for (int i = 0; i < topkDim; i++) {
            avg += originData[sortedIdx[i]];
        }
        double rClient = Math.abs(avg / topkDim);
        Boolean bClient = rangeReached == 0 ? rClient < (MULTIPLE * rEst) : rClient < (rEst / MULTIPLE);
        LOGGER.info("Actual r is" + rClient + " , actual b is " + bClient);
        return randomResponse(bClient, randomResponseEps);
    }

    /**
     * allocate privacy budget.
     *
     * @return privacy budget for magRR alg.
     */
    private static double allocateBudget() {
        final double RREPS = 5.0f;
        return RREPS;
    }

    /**
     * SignDS alg.
     *
     * @param client fl client
     * @param sign   random sign value.
     * @return bool and index list.
     */
    public Object[] signDSModel(Client client, boolean sign) {
        double randomResponseEps = allocateBudget();
        int layerNum = updateFeatureName.size();
        int inputDim = 0;
        for (int i = 0; i < layerNum; i++) {
            String key = updateFeatureName.get(i);
            float[] dataBeforeTrain = client.getPreFeature(key);
            inputDim += dataBeforeTrain.length;
        }
        int topkDim = (int) (signK * inputDim);
        if (signDimOut == 0) {
            signDimOut = findOptOutputDim(signThrRatio, topkDim, inputDim, signEps);
        }
        int thrDim = calcOptThr(topkDim, inputDim, signDimOut, signEps);
        double combLessInter = 0d;
        double combMoreInter = 0d;
        for (int i = 0; i < thrDim; i++) {
            combLessInter += countCombs(i, topkDim, inputDim, signDimOut);
        }
        for (int i = thrDim; i <= signDimOut; i++) {
            combMoreInter += countCombs(i, topkDim, inputDim, signDimOut);
        }
        double denominator = combLessInter + Math.exp(signEps) * combMoreInter;
        if (denominator == 0) {
            LOGGER.severe("[SignDS] denominator is 0, please check");
            return new Object[]{true, new int[0]};
        }
        int numInter = countInters(thrDim, denominator, topkDim, inputDim, signDimOut, signEps);
        LOGGER.info("[SignDS] numInter is " + numInter);
        int numOuter = signDimOut - numInter;
        if (topkDim < numInter || signDimOut <= 0) {
            LOGGER.severe("[SignDS] topkDim or signDimOut is ERROR! please check");
            return new Object[]{true, new int[0]};
        }

        float[] originData = new float[inputDim];
        int index = 0;
        for (int i = 0; i < layerNum; i++) {
            String key = updateFeatureName.get(i);
            float[] dataAfterTrain = client.getFeature(key);
            float[] dataBeforeTrain = client.getPreFeature(key);
            for (int j = 0; j < dataAfterTrain.length; j++) {
                float updateData = dataAfterTrain[j] - dataBeforeTrain[j];
                originData[index] = updateData;
                index++;
            }
        }

        int[] sortedIdx = mergeShort(originData, sign);
        boolean bHat = magRR(originData, sortedIdx, topkDim, localFLParameter.getREst(), randomResponseEps,
                localFLParameter.getRangeReached());
        int[] outputDimensionIndexList = new int[signDimOut];
        SecureRandom secureRandom = Common.getSecureRandom();
        randomSelect(secureRandom, sortedIdx, 0, topkDim, numInter, outputDimensionIndexList, 0);
        randomSelect(secureRandom, sortedIdx, topkDim, inputDim - topkDim,
                numOuter, outputDimensionIndexList, numInter);
        Arrays.sort(outputDimensionIndexList);
        LOGGER.info("[SignDS] outputDimension size is " + outputDimensionIndexList.length);
        return new Object[]{bHat, outputDimensionIndexList};
    }

    /**
     * generate laplace noise.
     *
     * @param secureRandom
     * @param beta
     * @return
     */
    float genLaplaceNoise(SecureRandom secureRandom, float beta) {
        float u1 = secureRandom.nextFloat();
        float u2 = secureRandom.nextFloat();
        int tryTimeLimit = 100;
        int tryTimeCount = 0;
        while (u2 == 0) {
            u2 = secureRandom.nextFloat();
            if (++tryTimeCount > tryTimeLimit) {
                return 0;
            }
        }
        if (u1 <= 0.5f) {
            return (float) (-beta * log(1. - u2));
        } else {
            return (float) (beta * log(u2));
        }
    }

    /**
     * add laplace noise to input data.
     *
     * @param data input data.
     * @param eps  privacy budget/
     * @return float list.
     */
    float[] addLaplaceNoise(float[] data, float eps) {
        if (eps <= 0 || eps > laplaceEpsUpper) {
            LOGGER.severe("eps for laplace is out of range.");
            return data;
        }
        if (data.length <= 0) {
            LOGGER.warning("The input data of laplace is empty, please check.");
            return data;
        }
        LOGGER.info("laplace eps is " + eps);
        float globalSensitivity = 1f;
        float beta = globalSensitivity / eps;
        SecureRandom secureRandom = Common.getSecureRandom();
        for (int i = 0; i < data.length; i++) {
            data[i] += genLaplaceNoise(secureRandom, beta);
        }
        return data;
    }
}
