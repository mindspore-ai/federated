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
import org.bouncycastle.math.ec.rfc7748.X25519;

import java.util.Map;
import java.util.logging.Logger;

/**
 * Defines global parameters used internally during federated learning.
 *
 * @since 2021-06-30
 */
public class LocalFLParameter {
    private static final Logger LOGGER = FLLoggerGenerater.getModelLogger(LocalFLParameter.class.toString());

    /**
     * Seed length used to generate random perturbations
     */
    public static final int SEED_SIZE = 32;

    /**
     * The length of IV value
     */
    public static final int I_VEC_LEN = 16;

    /**
     * The length of salt value
     */
    public static final int SALT_SIZE = 32;

    /**
     * the key length
     */
    public static final int KEY_LEN = X25519.SCALAR_SIZE;

    /**
     * The deployment environment supported by federated learning tasks: "android".
     */
    public static final String ANDROID = "android";

    /**
     * the type of the privacy protection scheme used in the eval result
     */
    public static final String NOT_EVAL = "NOT_EVAL";

    /**
     * the type of the privacy protection scheme used in the eval result
     */
    public static final String NOT_ENCRYPT = "NOT_ENCRYPT";

    /**
     * the type of the privacy protection scheme used in the eval result
     */
    public static final String LAPLACE_EVAL_TYPE = "LAPLACE";

    /**
     * The deployment environment supported by federated learning tasks: "x86".
     */
    public static final String X86 = "x86";
    private static volatile LocalFLParameter localFLParameter;
    private String flID;
    private String encryptLevel = EncryptLevel.NOT_ENCRYPT.toString();
    private String earlyStopMod = EarlyStopMod.NOT_EARLY_STOP.toString();
    private String serverMod = ServerMod.HYBRID_TRAINING.toString();
    private boolean stopJobFlag = false;
    private boolean useSSL = true;
    private float lr = 0.1f;
    private Map<String, float[]> oldFeatureMap;
    private byte uploadCompressType = 0;
    private int seed = 0;
    private float uploadSparseRatio = 0.08f;

    // default DeviceType:DT_CPU -> 0
    private int deviceType = 0;
    // default ThreadNum -> 2
    private int threadNum = 2;
    // default cpuBindMode:NO_BIND ->  0
    private int cpuBindMode = 0;
    // default enable_fp16 -> false
    private boolean enableFp16 = false;

    // default unsupervised train evaluate enable/disable flg.
    private String unsupervisedEvalFlg = NOT_ENCRYPT;

    // default rEst -> 0.00001f
    private float rEst = 0.00001f;
    // default rangeReached -> false
    private int rangeReached = 0;

    private LocalFLParameter() {
    }

    /**
     * Get the singleton object of the class LocalFLParameter.
     *
     * @return the singleton object of the class LocalFLParameter.
     */
    public static LocalFLParameter getInstance() {
        LocalFLParameter localRef = localFLParameter;
        if (localRef == null) {
            synchronized (LocalFLParameter.class) {
                localRef = localFLParameter;
                if (localRef == null) {
                    localFLParameter = localRef = new LocalFLParameter();
                }
            }
        }
        return localRef;
    }

    public String getFlID() {
        if (flID == null || flID.isEmpty()) {
            LOGGER.severe("[localFLParameter] the parameter of <flID> is null, please set it before " +
                    "use");
            throw new IllegalArgumentException();
        }
        return flID;
    }

    public void setFlID(String flID) {
        if (flID == null || flID.isEmpty()) {
            LOGGER.severe("[localFLParameter] the parameter of <flID> is null, please check it before " +
                    "set");
            throw new IllegalArgumentException();
        }
        this.flID = flID;
    }

    public EncryptLevel getEncryptLevel() {
        return EncryptLevel.valueOf(encryptLevel);
    }

    public void setEncryptLevel(String encryptLevel) {
        if (encryptLevel == null || encryptLevel.isEmpty()) {
            LOGGER.severe("[localFLParameter] the parameter of <encryptLevel> is null, please check it " +
                    "before setting");
            throw new IllegalArgumentException();
        }
        if ((!EncryptLevel.DP_ENCRYPT.toString().equals(encryptLevel)) &&
                (!EncryptLevel.NOT_ENCRYPT.toString().equals(encryptLevel)) &&
                (!EncryptLevel.SIGNDS.toString().equals(encryptLevel)) &&
                (!EncryptLevel.PW_ENCRYPT.toString().equals(encryptLevel))) {
            LOGGER.severe("[localFLParameter] the parameter of <encryptLevel> is " + encryptLevel + " ," +
                    " it must be DP_ENCRYPT or NOT_ENCRYPT or PW_ENCRYPT or SIGNDS, please check it before setting");
            throw new IllegalArgumentException();
        }
        this.encryptLevel = encryptLevel;
    }

    public EarlyStopMod getEarlyStopMod() {
        return EarlyStopMod.valueOf(earlyStopMod);
    }

    public void setEarlyStopMod(String earlyStopMod) {
        if (earlyStopMod == null || earlyStopMod.isEmpty()) {
            LOGGER.severe("[localFLParameter] the parameter of <earlyStopMod> is null, please check it " +
                    "before setting");
            throw new IllegalArgumentException();
        }
        if ((!EarlyStopMod.NOT_EARLY_STOP.toString().equals(earlyStopMod)) &&
                (!EarlyStopMod.LOSS_ABS.toString().equals(earlyStopMod)) &&
                (!EarlyStopMod.LOSS_DIFF.toString().equals(earlyStopMod)) &&
                (!EarlyStopMod.WEIGHT_DIFF.toString().equals(earlyStopMod))) {
            LOGGER.severe("[localFLParameter] the parameter of <earlyStopMod> is " + earlyStopMod + " ," +
                    " it must be NOT_EARLY_STOP or LOSS_ABS or LOSS_DIFF or WEIGHT_DIFF, please check it before " +
                    "setting");
            throw new IllegalArgumentException();
        }
        this.earlyStopMod = earlyStopMod;
    }

    public String getServerMod() {
        return serverMod;
    }

    public void setServerMod(String serverMod) {
        if (serverMod == null || serverMod.isEmpty()) {
            LOGGER.severe("[localFLParameter] the parameter of <serverMod> is null, please check it " +
                    "before setting");
            throw new IllegalArgumentException();
        }
        if ((!ServerMod.HYBRID_TRAINING.toString().equals(serverMod)) &&
                (!ServerMod.FEDERATED_LEARNING.toString().equals(serverMod))) {
            LOGGER.severe("[localFLParameter] the parameter of <serverMod> is " + serverMod + " , it " +
                    "must be HYBRID_TRAINING or FEDERATED_LEARNING, please check it before setting");
            throw new IllegalArgumentException();
        }
        this.serverMod = serverMod;
    }

    public boolean isStopJobFlag() {
        return stopJobFlag;
    }

    public void setStopJobFlag(boolean stopJobFlag) {
        this.stopJobFlag = stopJobFlag;
    }


    public int getDeviceType() {
        return deviceType;
    }

    public int getThreadNum() {
        return threadNum;
    }

    public int getCpuBindMode() {
        return cpuBindMode;
    }

    public boolean isEnableFp16() {
        return enableFp16;
    }

    public void setMsConfig(int deviceType, int threadNum, int cpuBindMode, boolean enableFp16) {
        this.deviceType = deviceType;
        this.threadNum = threadNum;
        this.cpuBindMode = cpuBindMode;
        this.enableFp16 = enableFp16;
    }

    public boolean isUseSSL() {
        return useSSL;
    }

    public void setUseSSL(boolean useSSL) {
        this.useSSL = useSSL;
    }

    public float getLr() {
        return lr;
    }

    public void setLr(float lr) {
        this.lr = lr;
    }

    public byte getUploadCompressType() {
        return uploadCompressType;
    }

    public void setUploadCompressType(byte uploadCompressType) {
        this.uploadCompressType = uploadCompressType;
    }

    public int getSeed() {
        return seed;
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }

    public float getUploadSparseRatio() {
        return uploadSparseRatio;
    }

    public void setUploadSparseRatio(float uploadSparseRatio) {
        this.uploadSparseRatio = uploadSparseRatio;
    }

    public String getUnsupervisedEvalFlg() {
        return unsupervisedEvalFlg;
    }

    public void setUnsupervisedEvalFlg(String unsupervisedEvalFlg) {
        this.unsupervisedEvalFlg = unsupervisedEvalFlg;
    }

    public float getREst() {
        return rEst;
    }

    public void setREst(float rEstServer) {
        if (rEstServer >= 0 && rEstServer < 1) {
            rEst = rEstServer;
        } else {
            LOGGER.warning("[flParameter] the parameter of <rEst> is " + rEstServer + "use default value: 1");
            rEst = 1;
        }
    }

    public int getRangeReached() {
        return rangeReached;
    }

    public void setRangeReached(int rangeReachedServer) {
        rangeReached = rangeReachedServer;
    }
}
