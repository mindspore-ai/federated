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

import static com.mindspore.flclient.FLParameter.TIME_OUT;
import static com.mindspore.flclient.LocalFLParameter.ANDROID;
import static com.mindspore.flclient.LocalFLParameter.X86;

import com.mindspore.flclient.common.FLLoggerGenerater;
import okhttp3.*;

import java.io.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.logging.Logger;

import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.X509TrustManager;

/**
 * Define the communication interface.
 *
 * @since 2021-06-30
 */
public class FLCommunication implements IFLCommunication {
    private static int timeOut;
    private static String sslProtocol;
    private static String env;
    private static SSLSocketFactory sslSocketFactory;
    private static X509TrustManager x509TrustManager;
    private static final MediaType MEDIA_TYPE_JSON = MediaType.parse("applicatiom/json;charset=utf-8");
    private static final Logger LOGGER = FLLoggerGenerater.getModelLogger(FLCommunication.class.toString());
    private static volatile FLCommunication communication;
    private static boolean msgDumpFlg = false;
    private static String msgDumpPath;
    private int msgDumpIdx = 0;

    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private OkHttpClient client;

    private FLCommunication() {
        if (flParameter.getTimeOut() != 0) {
            timeOut = flParameter.getTimeOut();
        } else {
            timeOut = TIME_OUT;
        }
        sslProtocol = flParameter.getSslProtocol();
        env = flParameter.getDeployEnv();
        if (ANDROID.equals(env) && Common.isHttps()) {
            sslSocketFactory = flParameter.getSslSocketFactory();
            x509TrustManager = flParameter.getX509TrustManager();
        }
        client = getOkHttpClient();
    }

    public static void setMsgDumpFlg(boolean msgDumpFlg) {
        FLCommunication.msgDumpFlg = msgDumpFlg;
    }
    public static void setMsgDumpPath(String msgDumpPath) {
        FLCommunication.msgDumpPath = msgDumpPath;
    }

    private static OkHttpClient getOkHttpClient() {
        LOGGER.info("Set timeOut in OkHttpClient: " + timeOut);
        OkHttpClient.Builder builder = new OkHttpClient.Builder();
        builder.connectTimeout(timeOut, TimeUnit.SECONDS);
        builder.writeTimeout(timeOut, TimeUnit.SECONDS);
        builder.readTimeout(3 * timeOut, TimeUnit.SECONDS);
        if(!Common.isHttps()){
            LOGGER.info("conducting http communication, do not need SSLSocketFactoryTools");
            return builder.build();
        }

        if (ANDROID.equals(env)) {
            builder.sslSocketFactory(sslSocketFactory, x509TrustManager);
        } else {
            builder.sslSocketFactory(SSLSocketFactoryTools.getInstance().getmSslSocketFactory(),
                    SSLSocketFactoryTools.getInstance().getmTrustManager());
        }
        return builder.build();
    }

    /**
     * Get the singleton object of the class FLCommunication.
     *
     * @return the singleton object of the class FLCommunication.
     */
    public static FLCommunication getInstance() {
        FLCommunication localRef = communication;
        if (localRef == null) {
            synchronized (FLCommunication.class) {
                localRef = communication;
                if (localRef == null) {
                    communication = localRef = new FLCommunication();
                }
            }
        }
        return localRef;
    }

    @Override
    public void setTimeOut(int timeout) throws TimeoutException {
    }

    private void dumpMsgBodyToFile(String url, byte[] reqBody, byte[] resBody) {
        String urlPath[] = url.split("/");
        String realPath = msgDumpPath + '/' + urlPath[urlPath.length - 1];
        String reqFileName = realPath + "/Req_" + Integer.toString(msgDumpIdx);
        String resFileName = realPath + "/Res_" + Integer.toString(msgDumpIdx);
        try {
            File dir = new File(realPath);
            if (!dir.exists()) {
                dir.mkdir();
            }
            ObjectOutputStream oosReq = new ObjectOutputStream(new FileOutputStream(reqFileName));
            ObjectOutputStream oosRes = new ObjectOutputStream(new FileOutputStream(resFileName));
            oosReq.writeObject(reqBody);
            oosRes.writeObject(resBody);
            oosReq.close();
            oosRes.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        msgDumpIdx++;
    }

    @Override
    public byte[] syncRequest(String url, byte[] msg) throws IOException {
        Request request = new Request.Builder()
                .url(url)
                .post(RequestBody.create(MEDIA_TYPE_JSON, msg)).build();
        Response response = this.client.newCall(request).execute();
        if (!response.isSuccessful()) {
            throw new IOException("Unexpected code " + response);
        }
        if (response.body() == null) {
            throw new IOException("the returned response is null");
        }
        byte[] responseBody = response.body().bytes();
        if (msgDumpFlg) {
            dumpMsgBodyToFile(url, msg, responseBody);
        }
        return responseBody;
    }

    @Override
    public void asyncRequest(String url, byte[] msg, IAsyncCallBack callBack) throws Exception {
        Request request = new Request.Builder()
                .url(url)
                .header("Accept", "application/proto")
                .header("Content-Type", "application/proto; charset=utf-8")
                .post(RequestBody.create(MEDIA_TYPE_JSON, msg)).build();

        client.newCall(request).enqueue(new Callback() {
            IAsyncCallBack asyncCallBack = callBack;

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                asyncCallBack.onResponse(response.body().bytes());
                call.cancel();
            }

            @Override
            public void onFailure(Call call, IOException ioException) {
                asyncCallBack.onFailure(ioException);
                call.cancel();
            }
        });
    }
}
