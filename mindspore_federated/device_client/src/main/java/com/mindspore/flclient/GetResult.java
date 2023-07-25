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

package com.mindspore.flclient;

import com.google.flatbuffers.FlatBufferBuilder;
import com.mindspore.flclient.common.FLLoggerGenerater;
import mindspore.fl.schema.*;

import java.util.Date;
import java.util.logging.Logger;

/**
 * Define the serialization method, handle the response message returned from server for getResult request.
 */
public class GetResult {
    private static final Logger LOGGER = FLLoggerGenerater.getModelLogger(GetResult.class.toString());
    private static volatile GetResult getResult;

    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private int retCode = ResponseCode.RequestError;

    private GetResult() {
    }

    /**
     * Get the singleton object of the class GetResult.
     *
     * @return the singleton object of the class GetResult.
     */
    public static GetResult getInstance() {
        GetResult localRef = getResult;
        if (localRef == null) {
            synchronized (GetResult.class) {
                localRef = getResult;
                if (localRef == null) {
                    getResult = localRef = new GetResult();
                }
            }
        }
        return localRef;
    }

    public int getRetCode() {
        return retCode;
    }

    /**
     * Get a flatBuffer builder of RequestGetResult.
     *
     * @param name      the model name.
     * @param iteration current iteration of federated learning task.
     * @return the flatBuffer builder of RequestGetResult in byte[] format.
     */
    public byte[] getRequestGetResult(String name, int iteration) {
        if (name == null || name.isEmpty()) {
            LOGGER.severe("[GetResult] the input parameter of <name> is null or empty, please check!");
            throw new IllegalArgumentException();
        }
        RequestGetResultBuilder builder = new RequestGetResultBuilder();
        return builder.iteration(iteration).flName(name).time().build();
    }

    /**
     * Handle the response message returned from server.
     *
     * @param responseDataBuf the response message returned from server.
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus doResponse(ResponseGetResult responseDataBuf) {
        retCode = responseDataBuf.retcode();
        LOGGER.info("[getResult] ==========the response message of getResult is:================");
        LOGGER.info("[getResult] ==========retCode: " + retCode);
        LOGGER.info("[getResult] ==========reason: " + responseDataBuf.reason());
        LOGGER.info("[getResult] ==========iteration: " + responseDataBuf.iteration());
        LOGGER.info("[getResult] ==========time: " + responseDataBuf.timestamp());
        FLClientStatus status = FLClientStatus.SUCCESS;
        switch (responseDataBuf.retcode()) {
            case (ResponseCode.SUCCEED):
                LOGGER.info("[getResult] SUCCESS");
                return status;
            case (ResponseCode.SucNotReady):
                LOGGER.info("[getResult] server is not ready now: need wait and request getResult again");
                return FLClientStatus.WAIT;
            case (ResponseCode.OutOfTime):
                LOGGER.info("[getResult] out of time: need wait and request startFLJob again");
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.warning("[getResult] catch RequestError or SystemError");
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe("[getResult] the return <retCode> from server is invalid: " + retCode);
                return FLClientStatus.FAILED;
        }
    }

    class RequestGetResultBuilder {
        private FlatBufferBuilder builder;
        private int nameOffset = 0;
        private int iteration = 0;
        private int timeStampOffset = 0;

        public RequestGetResultBuilder() {
            builder = new FlatBufferBuilder();
        }

        private RequestGetResultBuilder flName(String name) {
            if (name == null || name.isEmpty()) {
                LOGGER.severe("[GetResult] the input parameter of <name> is null or empty, please " +
                        "check!");
                throw new IllegalArgumentException();
            }
            this.nameOffset = this.builder.createString(name);
            return this;
        }

        private RequestGetResultBuilder time() {
            Date date = new Date();
            long time = date.getTime();
            this.timeStampOffset = builder.createString(String.valueOf(time));
            return this;
        }

        private RequestGetResultBuilder iteration(int iteration) {
            this.iteration = iteration;
            return this;
        }

        private byte[] build() {
            RequestGetResult.startRequestGetResult(builder);
            RequestGetResult.addFlName(builder, nameOffset);
            RequestGetResult.addIteration(builder, iteration);
            RequestGetResult.addTimestamp(builder, timeStampOffset);
            int root = RequestGetResult.endRequestGetResult(builder);
            builder.finish(root);
            return builder.sizedByteArray();
        }
    }
}
