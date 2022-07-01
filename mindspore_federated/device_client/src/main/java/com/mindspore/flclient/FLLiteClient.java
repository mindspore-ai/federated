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

import com.mindspore.flclient.common.FLLoggerGenerater;
import com.mindspore.flclient.model.Client;
import com.mindspore.flclient.model.ClientManager;
import com.mindspore.flclient.model.CommonUtils;
import com.mindspore.flclient.model.RunType;
import com.mindspore.flclient.model.Status;
import com.mindspore.flclient.pki.PkiBean;
import com.mindspore.flclient.pki.PkiUtil;

import mindspore.fl.schema.CipherPublicParams;
import mindspore.fl.schema.FLPlan;
import mindspore.fl.schema.ResponseCode;
import mindspore.fl.schema.ResponseFLJob;
import mindspore.fl.schema.ResponseGetModel;
import mindspore.fl.schema.ResponseUpdateModel;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import java.util.ArrayList;

/**
 * Defining the general process of federated learning tasks.
 *
 * @since 2021-06-30
 */
public class FLLiteClient {
    private static final Logger LOGGER = FLLoggerGenerater.getModelLogger(FLLiteClient.class.toString());
    private static int iteration = 0;
    private static Map<String, float[]> mapBeforeTrain;

    private double dpNormClipFactor = 1.0d;
    private double dpNormClipAdapt = 0.05d;
    private FLCommunication flCommunication;
    private FLClientStatus status;
    private int retCode = ResponseCode.RequestError;
    private int iterations = 1;
    private int epochs = 1;
    private int batchSize = 16;
    private int minSecretNum;
    private byte[] prime;
    private int featureSize;
    private int trainDataSize;
    private double dpEps = 100d;
    private double dpDelta = 0.01d;
    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private SecureProtocol secureProtocol = new SecureProtocol();
    private String nextRequestTime;
    private Client client;
    private float signK = 0.01f;
    private float signEps = 100;
    private float signThrRatio = 0.6f;
    private float signGlobalLr = 1f;
    private int signDimOut = 0;

    /**
     * Defining a constructor of teh class FLLiteClient.
     */
    public FLLiteClient() {
        flCommunication = FLCommunication.getInstance();
        client = ClientManager.getClient(flParameter.getFlName());
    }

    private int setGlobalParameters(ResponseFLJob flJob) {
        FLPlan flPlan = flJob.flPlanConfig();
        if (flPlan == null) {
            LOGGER.severe("[startFLJob] the FLPlan get from server is null");
            return -1;
        }
        iterations = flPlan.iterations();
        epochs = flPlan.epochs();
        batchSize = flPlan.miniBatch();
        String serverMod = flPlan.serverMode();
        localFLParameter.setServerMod(serverMod);
        // Get and set hyper parameters for compression.
        byte uploadCompressType = flJob.uploadCompressType();
        LOGGER.info("[startFLJob] [compression] uploadCompressType: " + uploadCompressType);
        localFLParameter.setUploadCompressType(uploadCompressType);
        float uploadSparseRate = flJob.uploadSparseRate();
        LOGGER.info("[startFLJob] [compression] uploadSparseRate: " + uploadSparseRate);
        localFLParameter.setUploadSparseRatio(uploadSparseRate);
        int seed = flJob.iteration();
        LOGGER.info("[startFLJob] [compression] seed: " + seed);
        localFLParameter.setSeed(seed);
        LOGGER.info("[startFLJob] the GlobalParameter <serverMod> from server: " + serverMod);
        LOGGER.info("[startFLJob] the GlobalParameter <iterations> from server: " + iterations);
        LOGGER.info("[startFLJob] the GlobalParameter <epochs> from server: " + epochs);
        LOGGER.info("[startFLJob] the GlobalParameter <batchSize> from server: " + batchSize);
        CipherPublicParams cipherPublicParams = flPlan.cipher();
        if (cipherPublicParams == null) {
            LOGGER.severe("[startFLJob] the cipherPublicParams returned from server is null");
            return -1;
        }
        String encryptLevel = cipherPublicParams.encryptType();
        if (encryptLevel == null || encryptLevel.isEmpty()) {
            LOGGER.severe("[startFLJob] GlobalParameters <encryptLevel> from server is null, set the " +
                    "encryptLevel to NOT_ENCRYPT ");
            localFLParameter.setEncryptLevel(EncryptLevel.NOT_ENCRYPT.toString());
        } else {
            localFLParameter.setEncryptLevel(encryptLevel);
            LOGGER.info("[startFLJob] GlobalParameters <encryptLevel> from server: " + encryptLevel);
        }
        switch (localFLParameter.getEncryptLevel()) {
            case PW_ENCRYPT:
                minSecretNum = cipherPublicParams.pwParams().t();
                int primeLength = cipherPublicParams.pwParams().primeLength();
                prime = new byte[primeLength];
                for (int i = 0; i < primeLength; i++) {
                    prime[i] = (byte) cipherPublicParams.pwParams().prime(i);
                }
                LOGGER.info("[startFLJob] GlobalParameters <minSecretNum> from server: " + minSecretNum);
                if (minSecretNum <= 0) {
                    LOGGER.info("[startFLJob] GlobalParameters <minSecretNum> from server is not valid:" +
                            "  <=0");
                    return -1;
                }
                break;
            case DP_ENCRYPT:
                dpEps = cipherPublicParams.dpParams().dpEps();
                dpDelta = cipherPublicParams.dpParams().dpDelta();
                dpNormClipFactor = cipherPublicParams.dpParams().dpNormClip();
                LOGGER.info("[startFLJob] GlobalParameters <dpEps> from server: " + dpEps);
                LOGGER.info("[startFLJob] GlobalParameters <dpDelta> from server: " + dpDelta);
                LOGGER.info("[startFLJob] GlobalParameters <dpNormClipFactor> from server: " +
                        dpNormClipFactor);
                break;
            case SIGNDS:
                signK = cipherPublicParams.dsParams().signK();
                signEps = cipherPublicParams.dsParams().signEps();
                signThrRatio = cipherPublicParams.dsParams().signThrRatio();
                signGlobalLr = cipherPublicParams.dsParams().signGlobalLr();
                signDimOut = cipherPublicParams.dsParams().signDimOut();
                LOGGER.info("[startFLJob] GlobalParameters <signK> from server: " + signK);
                LOGGER.info("[startFLJob] GlobalParameters <signEps> from server: " + signEps);
                LOGGER.info("[startFLJob] GlobalParameters <signThrRatio> from server: " + signThrRatio);
                LOGGER.info("[startFLJob] GlobalParameters <signGlobalLr> from server: " + signGlobalLr);
                LOGGER.info("[startFLJob] GlobalParameters <SignDimOut> from server: " + signDimOut);
                break;
            default:
                LOGGER.info("[startFLJob] NOT_ENCRYPT, do not set parameter for Encrypt");
        }
        return 0;
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
     * Obtain current iteration returned by server.
     *
     * @return the current iteration returned by server.
     */
    public int getIteration() {
        return iteration;
    }

    /**
     * Obtain total iterations for the task returned by server.
     *
     * @return the total iterations for the task returned by server.
     */
    public int getIterations() {
        return iterations;
    }

    /**
     * Obtain the returned timestamp for next request from server.
     *
     * @return the timestamp for next request.
     */
    public String getNextRequestTime() {
        return nextRequestTime;
    }

    /**
     * set the size of train date set.
     *
     * @param trainDataSize the size of train date set.
     */
    public void setTrainDataSize(int trainDataSize) {
        this.trainDataSize = trainDataSize;
    }

    /**
     * Obtain the dpNormClipFactor.
     *
     * @return the dpNormClipFactor.
     */
    public double getDpNormClipFactor() {
        return dpNormClipFactor;
    }

    /**
     * Obtain the dpNormClipAdapt.
     *
     * @return the dpNormClipAdapt.
     */
    public double getDpNormClipAdapt() {
        return dpNormClipAdapt;
    }

    /**
     * Set the dpNormClipAdapt.
     *
     * @param dpNormClipAdapt the dpNormClipAdapt.
     */
    public void setDpNormClipAdapt(double dpNormClipAdapt) {
        this.dpNormClipAdapt = dpNormClipAdapt;
    }

    /**
     * Send serialized request message of startFLJob to server.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus startFLJob() {
        LOGGER.info("[startFLJob] ====================================Verify " +
                "server====================================");
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(),
                flParameter.getDomainName());
        StartFLJob startFLJob = StartFLJob.getInstance();
        Date date = new Date();
        long time = date.getTime();

        PkiBean pkiBean = null;
        if (flParameter.isPkiVerify()) {
            pkiBean = PkiUtil.genPkiBean(flParameter.getClientID(), time);
        }
        byte[] msg = startFLJob.getRequestStartFLJob(trainDataSize, iteration, time, pkiBean);
        try {
            long start = Common.startTime("single startFLJob");
            LOGGER.info("[startFLJob] the request message length: " + msg.length);
            byte[] message = flCommunication.syncRequest(url + "/startFLJob", msg);
            if (!Common.isSeverReady(message)) {
                LOGGER.info("[startFLJob] the server is not ready now, need wait some time and request " +
                        "again");
                status = FLClientStatus.RESTART;
                nextRequestTime = Common.getNextReqTime();
                retCode = ResponseCode.OutOfTime;
                return status;
            }
            if (Common.isSeverJobFinished(message)) {
                return serverJobFinished("startFLJob");
            }
            LOGGER.info("[startFLJob] the response message length: " + message.length);
            Common.endTime(start, "single startFLJob");
            ByteBuffer buffer = ByteBuffer.wrap(message);
            ResponseFLJob responseDataBuf = ResponseFLJob.getRootAsResponseFLJob(buffer);
            status = judgeStartFLJob(startFLJob, responseDataBuf);
        } catch (IOException e) {
            failed("[startFLJob] unsolved error code in StartFLJob: catch IOException: " + e.getMessage(),
                    ResponseCode.RequestError);
        }
        return status;
    }

    private FLClientStatus judgeStartFLJob(StartFLJob startFLJob, ResponseFLJob responseDataBuf) {
        iteration = responseDataBuf.iteration();
        FLClientStatus response = startFLJob.doResponse(responseDataBuf);
        retCode = startFLJob.getRetCode();
        status = response;
        switch (response) {
            case SUCCESS:
                LOGGER.info("[startFLJob] startFLJob success");
                featureSize = startFLJob.getFeatureSize();
                secureProtocol.setUpdateFeatureName(startFLJob.getUpdateFeatureName());
                LOGGER.info("[startFLJob] ***the feature size get in ResponseFLJob***: " + featureSize);
                int tag = setGlobalParameters(responseDataBuf);
                if (tag == -1) {
                    LOGGER.severe("[startFLJob] setGlobalParameters failed");
                    status = FLClientStatus.FAILED;
                }
                break;
            case RESTART:
                FLPlan flPlan = responseDataBuf.flPlanConfig();
                if (flPlan == null) {
                    LOGGER.severe("[startFLJob] the flPlan returned from server is null");
                    return FLClientStatus.FAILED;
                }
                iterations = flPlan.iterations();
                LOGGER.info("[startFLJob] GlobalParameters <iterations> from server: " + iterations);
                nextRequestTime = responseDataBuf.nextReqTime();
                break;
            case FAILED:
                LOGGER.severe("[startFLJob] startFLJob failed");
                break;
            default:
                LOGGER.severe("[startFLJob] failed: the response of startFLJob is out of range " +
                        "<SUCCESS, WAIT, FAILED, Restart>");
                status = FLClientStatus.FAILED;
        }
        return status;
    }

    private FLClientStatus trainLoop() {
        Client client = ClientManager.getClient(flParameter.getFlName());
        if(!client.EnableTrain(true)){
            retCode = ResponseCode.RequestError;
            return FLClientStatus.FAILED;
        }
        retCode = ResponseCode.SUCCEED;
        LOGGER.info("[train] train in " + flParameter.getFlName());
        LOGGER.info("[train] lr for client is: " + localFLParameter.getLr());
        Status tag = client.setLearningRate(localFLParameter.getLr());
        if (!Status.SUCCESS.equals(tag)) {
            LOGGER.severe("[train] setLearningRate failed, return -1, please check");
            retCode = ResponseCode.RequestError;
            return FLClientStatus.FAILED;
        }
        tag = client.trainModel(epochs);
        if (Float.isNaN(client.getUploadLoss()) || Float.isInfinite(client.getUploadLoss())) {
            client.restoreModelFile(flParameter.getTrainModelPath());
            failed("[train] train failed, train loss is:" + client.getUploadLoss(), ResponseCode.RequestError);
        } else if (!Status.SUCCESS.equals(tag)) {
            failed("[train] unsolved error code in <client.trainModel>", ResponseCode.RequestError);
        }
        return status;
    }

    /**
     * Define the training process.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus localTrain() {
        LOGGER.info("[train] ====================================global train epoch " + iteration +
                "====================================");
        status = trainLoop();
        return status;
    }

    /**
     * Send serialized request message of updateModel to server.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus updateModel() {
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(),
                flParameter.getDomainName());
        UpdateModel updateModelBuf = UpdateModel.getInstance();
        byte[] updateModelBuffer = updateModelBuf.getRequestUpdateFLJob(iteration, secureProtocol, trainDataSize);
        if (updateModelBuf.getStatus() == FLClientStatus.FAILED) {
            LOGGER.info("[updateModel] catch error in build RequestUpdateFLJob");
            return FLClientStatus.FAILED;
        }
        try {
            long start = Common.startTime("single updateModel");
            LOGGER.info("[updateModel] the request message length: " + updateModelBuffer.length);
            byte[] message = flCommunication.syncRequest(url + "/updateModel", updateModelBuffer);
            if (!Common.isSeverReady(message)) {
                LOGGER.info("[updateModel] the server is not ready now, need wait some time and request" +
                        " again");
                status = FLClientStatus.RESTART;
                nextRequestTime = Common.getNextReqTime();
                retCode = ResponseCode.OutOfTime;
                return status;
            }
            if (Common.isSeverJobFinished(message)) {
                return serverJobFinished("updateModel");
            }
            LOGGER.info("[updateModel] the response message length: " + message.length);
            Common.endTime(start, "single updateModel");
            ByteBuffer debugBuffer = ByteBuffer.wrap(message);
            ResponseUpdateModel responseDataBuf = ResponseUpdateModel.getRootAsResponseUpdateModel(debugBuffer);
            status = updateModelBuf.doResponse(responseDataBuf);
            retCode = updateModelBuf.getRetCode();
            if (status == FLClientStatus.RESTART) {
                nextRequestTime = responseDataBuf.nextReqTime();
            }
            LOGGER.info("[updateModel] get response from server ok!");
        } catch (IOException e) {
            failed("[updateModel] unsolved error code in updateModel: catch IOException: " + e.getMessage(),
                    ResponseCode.RequestError);
        }
        return status;
    }

    /**
     * Send serialized request message of getModel to server.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus getModel() {
        String url = Common.generateUrl(flParameter.isUseElb(), flParameter.getServerNum(),
                flParameter.getDomainName());
        GetModel getModelBuf = GetModel.getInstance();
        byte[] buffer = getModelBuf.getRequestGetModel(flParameter.getFlName(), iteration);
        try {
            long start = Common.startTime("single getModel");
            LOGGER.info("[getModel] the request message length: " + buffer.length);
            byte[] message = flCommunication.syncRequest(url + "/getModel", buffer);
            if (!Common.isSeverReady(message)) {
                LOGGER.info("[getModel] the server is not ready now, need wait some time and request " +
                        "again");
                status = FLClientStatus.WAIT;
                retCode = ResponseCode.SucNotReady;
                return status;
            }
            if (Common.isSeverJobFinished(message)) {
                return serverJobFinished("getModel");
            }
            LOGGER.info("[getModel] the response message length: " + message.length);
            Common.endTime(start, "single getModel");
            LOGGER.info("[getModel] get model request success");
            ByteBuffer debugBuffer = ByteBuffer.wrap(message);
            ResponseGetModel responseDataBuf = ResponseGetModel.getRootAsResponseGetModel(debugBuffer);
            status = getModelBuf.doResponse(responseDataBuf);
            retCode = getModelBuf.getRetCode();
            if (status == FLClientStatus.RESTART) {
                nextRequestTime = responseDataBuf.timestamp();
            }
            LOGGER.info("[getModel] get response from server ok!");
        } catch (IOException e) {
            failed("[getModel] unsolved error code: catch IOException: " + e.getMessage(), ResponseCode.RequestError);
        }
        return status;
    }

    public void updateDpNormClip() {
        EncryptLevel encryptLevel = localFLParameter.getEncryptLevel();
        if (encryptLevel == EncryptLevel.DP_ENCRYPT) {
            client.EnableTrain(true);
            float fedWeightUpdateNorm = client.getDpWeightNorm(secureProtocol.getUpdateFeatureName());
            LOGGER.info("[DP] L2-norm of weights' average update is: " + fedWeightUpdateNorm);
            float newNormCLip = (float) getDpNormClipFactor() * fedWeightUpdateNorm;
            if (iteration == 1) {
                setDpNormClipAdapt(newNormCLip);
                LOGGER.info("[DP] dpNormClip has been updated.");
            } else {
                if (newNormCLip < getDpNormClipAdapt()) {
                    setDpNormClipAdapt(newNormCLip);
                    LOGGER.info("[DP] dpNormClip has been updated.");
                }
            }
            LOGGER.info("[DP] Adaptive dpNormClip is: " + getDpNormClipAdapt());
        }
    }

    /**
     * Obtain pairwise mask and individual mask.
     *
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus getFeatureMask() {
        FLClientStatus curStatus;
        switch (localFLParameter.getEncryptLevel()) {
            case PW_ENCRYPT:
                LOGGER.info("[Encrypt] creating feature mask of <" +
                        localFLParameter.getEncryptLevel().toString() + ">");
                secureProtocol.setPWParameter(iteration, minSecretNum, prime, featureSize);
                curStatus = secureProtocol.pwCreateMask();
                if (curStatus == FLClientStatus.RESTART) {
                    nextRequestTime = secureProtocol.getNextRequestTime();
                }
                retCode = secureProtocol.getRetCode();
                LOGGER.info("[Encrypt] the response of create mask for <" +
                        localFLParameter.getEncryptLevel().toString() + "> : " + curStatus);
                return curStatus;
            case DP_ENCRYPT:
                curStatus = secureProtocol.setDPParameter(iteration, dpEps, dpDelta, dpNormClipAdapt);
                retCode = ResponseCode.SUCCEED;
                if (curStatus != FLClientStatus.SUCCESS) {
                    LOGGER.severe("---Differential privacy init failed---");
                    retCode = ResponseCode.RequestError;
                    return FLClientStatus.FAILED;
                }
                LOGGER.info("[Encrypt] set parameters for DP_ENCRYPT!");
                return FLClientStatus.SUCCESS;
            case SIGNDS:
                curStatus = secureProtocol.setDSParameter(signK, signEps, signThrRatio, signGlobalLr, signDimOut);
                retCode = ResponseCode.SUCCEED;
                if (curStatus != FLClientStatus.SUCCESS) {
                    LOGGER.severe("---SignDS init failed---");
                    retCode = ResponseCode.RequestError;
                    return FLClientStatus.FAILED;
                }
                LOGGER.info("[Encrypt] set parameters for SignDS!");
                return FLClientStatus.SUCCESS;
            case NOT_ENCRYPT:
                retCode = ResponseCode.SUCCEED;
                LOGGER.info("[Encrypt] don't mask model");
                return FLClientStatus.SUCCESS;
            default:
                retCode = ResponseCode.SUCCEED;
                LOGGER.severe("[Encrypt] The encrypt level is error, not encrypt by default");
                return FLClientStatus.SUCCESS;
        }
    }

    /**
     * Reconstruct the secrets used for unmasking model weights.
     *
     * @return current status code in client.
     */
    public FLClientStatus unMasking() {
        FLClientStatus curStatus;
        switch (localFLParameter.getEncryptLevel()) {
            case PW_ENCRYPT:
                curStatus = secureProtocol.pwUnmasking();
                retCode = secureProtocol.getRetCode();
                LOGGER.info("[Encrypt] the response of unmasking : " + curStatus);
                if (curStatus == FLClientStatus.RESTART) {
                    nextRequestTime = secureProtocol.getNextRequestTime();
                }
                return curStatus;
            case DP_ENCRYPT:
                LOGGER.info("[Encrypt] DP_ENCRYPT do not need unmasking");
                retCode = ResponseCode.SUCCEED;
                return FLClientStatus.SUCCESS;
            case NOT_ENCRYPT:
                LOGGER.info("[Encrypt] haven't mask model");
                retCode = ResponseCode.SUCCEED;
                return FLClientStatus.SUCCESS;
            case SIGNDS:
                LOGGER.info("[Encrypt] SIGNDS do not need unmasking");
                retCode = ResponseCode.SUCCEED;
                return FLClientStatus.SUCCESS;
            default:
                LOGGER.severe("[Encrypt] The encrypt level is error, not encrypt by default");
                retCode = ResponseCode.SUCCEED;
                return FLClientStatus.SUCCESS;
        }
    }


    private FLClientStatus evaluateLoop() {
        status = FLClientStatus.SUCCESS;
        retCode = ResponseCode.SUCCEED;

        float acc = 0;
        if (localFLParameter.getServerMod().equals(ServerMod.HYBRID_TRAINING.toString())) {
            LOGGER.info("[evaluate] evaluateModel by " + localFLParameter.getServerMod());
            client.EnableTrain(false);
            LOGGER.info("[evaluate] modelPath: " + flParameter.getInferModelPath());
            acc = client.evalModel();
        } else {
            LOGGER.info("[evaluate] evaluateModel by " + localFLParameter.getServerMod());
            client.EnableTrain(true);
            LOGGER.info("[evaluate] modelPath: " + flParameter.getTrainModelPath());
            acc = client.evalModel();
        }
        if (Float.isNaN(acc)) {
            failed("[evaluate] unsolved error code in <evalModel>: the return acc is NAN", ResponseCode.RequestError);
            return status;
        }
        LOGGER.info("[evaluate] evaluate acc: " + acc);
        return status;
    }

    private void failed(String log, int retCode) {
        LOGGER.severe(log);
        status = FLClientStatus.FAILED;
        this.retCode = retCode;
    }

    /**
     * Evaluate model after getting model from server.
     *
     * @return the status code in client.
     */
    public FLClientStatus evaluateModel() {
        LOGGER.info("===================================evaluate model after getting model from " +
                "server===================================");
        status = evaluateLoop();
        return status;
    }

    /**
     * Set date path.
     *
     * @return date size.
     */
    public int setInput() {
        int dataSize = 0;
        retCode = ResponseCode.SUCCEED;
        LOGGER.info("==========set input===========");

        // train
        dataSize = client.initDataSets(flParameter.getDataMap()).get(RunType.TRAINMODE);
        if (dataSize <= 0) {
            retCode = ResponseCode.RequestError;
            return -1;
        }
        return dataSize;
    }

    private FLClientStatus serverJobFinished(String logTag) {
        LOGGER.info("[" + logTag + "] " + Common.JOB_NOT_AVAILABLE + " will stop the task and exist.");
        retCode = ResponseCode.SystemError;
        return FLClientStatus.FAILED;
    }
}
