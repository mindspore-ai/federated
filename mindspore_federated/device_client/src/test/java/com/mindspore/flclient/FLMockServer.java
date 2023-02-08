package com.mindspore.flclient;

import com.google.flatbuffers.FlatBufferBuilder;
import com.mindspore.MSTensor;
import com.mindspore.flclient.common.FLLoggerGenerater;
import com.mindspore.flclient.model.Client;
import com.mindspore.flclient.model.ClientManager;
import mindspore.fl.schema.*;
import okhttp3.mockwebserver.Dispatcher;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import okio.Buffer;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.logging.Logger;

import static mindspore.fl.schema.CompressType.NO_COMPRESS;
import static mindspore.fl.schema.UnsupervisedEvalFlg.EVAL_DISABLE;
import static mindspore.fl.schema.UnsupervisedEvalFlg.EVAL_ENABLE;

/**
 * Using this class to Mock the FL server node
 * @author       : zhangzhaoju
 * @since  : 2022/4/14
 */
class FLMockServer {
    private static final Logger LOGGER = FLLoggerGenerater.getModelLogger(FLMockServer.class.toString());
    private ArrayList<FLHttpRes> httpRes;
    private int httpResCnt = 0;
    private int curIter = 0;
    private int maxIter = 1;
    private MockWebServer server = new MockWebServer();
    private final Dispatcher dispatcher = new Dispatcher() {
        @Override
        public MockResponse dispatch(RecordedRequest request) throws InterruptedException {
            // Pop req from the request queue, avoid memory leak.
            server.takeRequest();
            if (httpRes == null) {
                LOGGER.severe("httpRes size is:" + Integer.toString(httpRes.size()) + " httpResCnt is:" + Integer.toString(httpResCnt));
                return new MockResponse().setResponseCode(404);
            }

            FLHttpRes curRes = httpRes.get(httpResCnt % httpRes.size());
            httpResCnt++;

            if(!reqMsgCheck(request, curRes)){
                return new MockResponse().setResponseCode(404);
            }

            if(curRes.getContendMode() != 0){
                return new MockResponse().setResponseCode(curRes.getResCode()).setBody(curRes.getContentData());
            }

            Buffer res = genResMsgBody(curRes);
            return new MockResponse().setResponseCode(curRes.getResCode()).setBody(res);
        }
    };

    public void setCaseRes(ArrayList<FLHttpRes> httpRes) {
        this.httpRes = httpRes;
        httpResCnt = 0;
    }

    private boolean reqMsgCheck(RecordedRequest request, FLHttpRes curRes){
        // check msg type
        if (!request.getPath().equals("/" + curRes.getResName())) {
            LOGGER.severe("The " + Integer.toString(httpResCnt) + "th expect msg is :" + curRes.getResName() + " but got " + request.getPath());
            return false;
        }
        // check msg content
        String msgName = curRes.getResName();
        if (msgName.equals("startFLJob")) {
            byte[] reqBody = request.getBody().readByteArray();
            ByteBuffer reqBuffer = ByteBuffer.wrap(reqBody);
            RequestFLJob reqFlJob = RequestFLJob.getRootAsRequestFLJob(reqBuffer);
            return true;
        }

        if (msgName.equals("updateModel")) {
            // do check for updateModel
            byte[] reqBody = request.getBody().readByteArray();
            ByteBuffer reqBuffer = ByteBuffer.wrap(reqBody);
            RequestUpdateModel reqUpdateModel = RequestUpdateModel.getRootAsRequestUpdateModel(reqBuffer);
            UnsupervisedEvalItems evalItems = reqUpdateModel.unsupervisedEvalItems();
            for(int i = 0; i < evalItems.evalItemsLength(); i++){
                UnsupervisedEvalItem evalItem = evalItems.evalItems(i);
                String evalName = evalItem.evalName();
                ByteBuffer byteBuffer = evalItem.evalDataAsByteBuffer();
                FloatBuffer floatBuffer =  byteBuffer.asFloatBuffer();
                float[] dst = new float[evalItem.evalDataLength()];
                floatBuffer.get(dst);
                LOGGER.info("Value of "+ evalName +" is:"+ Arrays.toString(dst));
            }
            return true;
        }

        if (msgName.equals("getModel")) {
            // do check for getModel
            return true;
        }

        LOGGER.severe("Got unsupported msg " + request.getPath());
        return false;
    }


    /**
     * Generate flat buffer data for CipherPublicParams
     * @param builder FlatBufferBuilder
     * @return the offset of CipherPublicParams
     */
    private int genCipherPublicParams(FlatBufferBuilder builder){
        builder.startTable(4);
        CipherPublicParams.addDsParams(builder, 0);
        CipherPublicParams.addDpParams(builder, 0);
        CipherPublicParams.addPwParams(builder, 0);
        CipherPublicParams.addEncryptType(builder, 0);
        return CipherPublicParams.endCipherPublicParams(builder);
    }

    /**
     * Generate flat buffer data for FLPlan
     * @param builder FlatBufferBuilder
     * @returnf the offset of FLPlan
     */
    private int genFLPlan(FlatBufferBuilder builder) {
        int fl_nameOffset = builder.createString("Lenet");
        int server_modeOffset = builder.createString("FEDERATED_LEARNING");
        int cipherOffset = genCipherPublicParams(builder);
        builder.startTable(11);
        FLPlan.addCipher(builder, cipherOffset);
        FLPlan.addMetrics(builder, 0);
        FLPlan.addAggregation(builder, 0);
        FLPlan.addLr(builder, 0.01f);
        FLPlan.addMiniBatch(builder, 32);
        FLPlan.addEarlyStop(builder, 0);
        FLPlan.addEpochs(builder, 3);
        FLPlan.addIterations(builder, 1);
        FLPlan.addFlName(builder, fl_nameOffset);
        FLPlan.addServerMode(builder, server_modeOffset);
        FLPlan.addShuffle(builder, false);
        return FLPlan.endFLPlan(builder);
    }

    /**
     * Generate flat buffer data for ResponseFLJob
     * @return the ByteBuffer of flat buffer
     */
    private byte[] genResponseFLJob() {
        FlatBufferBuilder builder = new FlatBufferBuilder();
        Date date = new Date();
        long curTimestamp = date.getTime();
        int curTimeOffset = builder.createString(String.valueOf(curTimestamp));
        int nextReqTimeOffset = builder.createString(String.valueOf(curTimestamp + 60 * 1000));

        int[] fmOffsets = getFmOffsets(builder);
        int featureMapOffset = ResponseFLJob.createFeatureMapVector(builder, fmOffsets);

        int flPlanConfigOffset = genFLPlan(builder);
        int reasonOffset = builder.createString("Success.");

        builder.startTable(13);
        ResponseFLJob.addCompressFeatureMap(builder, 0);
        ResponseFLJob.addUploadSparseRate(builder, 0.0f);
        ResponseFLJob.addTimestamp(builder, curTimeOffset);
        ResponseFLJob.addFeatureMap(builder, featureMapOffset);
        ResponseFLJob.addFlPlanConfig(builder, flPlanConfigOffset);
        ResponseFLJob.addNextReqTime(builder, nextReqTimeOffset);
        ResponseFLJob.addIteration(builder, 1);
        ResponseFLJob.addReason(builder, reasonOffset);
        ResponseFLJob.addRetcode(builder, 200);
        ResponseFLJob.addUnsupervisedEvalFlg(builder, EVAL_ENABLE);
        ResponseFLJob.addDownloadCompressType(builder, NO_COMPRESS);
        ResponseFLJob.addUploadCompressType(builder, NO_COMPRESS);
        ResponseFLJob.addIsSelected(builder, true);
        int root = ResponseFLJob.endResponseFLJob(builder);
        builder.finish(root);
        return builder.sizedByteArray();
    }

    private static int[] getFmOffsets(FlatBufferBuilder builder) {
        FLParameter flParameter = FLParameter.getInstance();
        Client client = ClientManager.getClient(flParameter.getFlName());
        client.EnableTrain(true);
        HashMap<String, MSTensor> feature_map = client.getAllFeature();
        int index = 0;
        int[] fmOffsets = new int[feature_map.size()];
        for (Map.Entry<String, MSTensor> item : feature_map.entrySet()) {
            String featureName = item.getKey();
            float[] featureData = item.getValue().getFloatData();
            LOGGER.fine("[updateModel build featuresMap] feature name: " + featureName + " feature " +
                    "size: " + featureData.length);
            int featureNameOffset = builder.createString(featureName);
            int weightOffset = FeatureMap.createDataVector(builder, featureData);
            int featureMapOffset = FeatureMap.createFeatureMap(builder, featureNameOffset, weightOffset);
            fmOffsets[index] = featureMapOffset;
            index += 1;
        }
        return fmOffsets;
    }

    /**
     * Generate flat buffer data for ResponseUpdateModel
     * @return the ByteBuffer of flat buffer
     */
    private byte[] genResponseUpdateModel() {
        FlatBufferBuilder builder = new FlatBufferBuilder();
        Date date = new Date();
        long curTimestamp = date.getTime();
        int curTimeOffset = builder.createString(String.valueOf(curTimestamp));
        int nextReqTimeOffset = builder.createString(String.valueOf(curTimestamp + 60 * 1000));
        int reasonOffset = builder.createString("Success.");

        builder.startTable(5);
        ResponseUpdateModel.addTimestamp(builder, curTimeOffset);
        ResponseUpdateModel.addNextReqTime(builder, nextReqTimeOffset);
        ResponseUpdateModel.addFeatureMap(builder, 0);
        ResponseUpdateModel.addReason(builder, reasonOffset);
        ResponseUpdateModel.addRetcode(builder, 200);
        int root = ResponseUpdateModel.endResponseUpdateModel(builder);
        builder.finish(root);
        return builder.sizedByteArray();
    }


    private byte[] genResponseGetModel() {
        FlatBufferBuilder builder = new FlatBufferBuilder();
        Date date = new Date();
        long curTimestamp = date.getTime();
        int curTimeOffset = builder.createString(String.valueOf(curTimestamp));
        int reasonOffset = builder.createString("Success.");
        int[] fmOffsets = getFmOffsets(builder);
        int featureMapOffset = ResponseGetModel.createFeatureMapVector(builder, fmOffsets);
        builder.startTable(7);
        ResponseGetModel.addCompressFeatureMap(builder, 0);
        ResponseGetModel.addTimestamp(builder, curTimeOffset);
        ResponseGetModel.addFeatureMap(builder, featureMapOffset);
        ResponseGetModel.addIteration(builder, 1);
        ResponseGetModel.addReason(builder, reasonOffset);
        ResponseGetModel.addRetcode(builder, 200);
        ResponseGetModel.addDownloadCompressType(builder, NO_COMPRESS);
        int root = ResponseGetModel.endResponseGetModel(builder);
        builder.finish(root);
        return builder.sizedByteArray();
    }

    private Buffer genResMsgBody(FLHttpRes curRes) {
        String msgName = curRes.getResName();
        if (msgName.equals("startFLJob")) {
            curIter++;
            byte[] msgBody = genResponseFLJob();
            ByteBuffer resBuffer = ByteBuffer.wrap(msgBody);
            ResponseFLJob responseDataBuf = ResponseFLJob.getRootAsResponseFLJob(resBuffer);
            responseDataBuf.flPlanConfig().mutateEpochs(1);  // change the flbuffer
            responseDataBuf.flPlanConfig().mutateIterations(maxIter); // only hase one iteration
            responseDataBuf.mutateIteration(curIter); // cur iteration
            Buffer buffer = new Buffer();
            buffer.write(responseDataBuf.getByteBuffer().array());
            return buffer;
        }

        if (msgName.equals("updateModel")) {
            byte[] msgBody = genResponseUpdateModel();
            Buffer buffer = new Buffer();
            ByteBuffer resBuffer = ByteBuffer.wrap(msgBody);
            ResponseUpdateModel responseDataBuf = ResponseUpdateModel.getRootAsResponseUpdateModel(resBuffer);
            buffer.write(responseDataBuf.getByteBuffer().array());
            return buffer;
        }

        if (msgName.equals("getModel")) {
            byte[] msgBody = genResponseGetModel();
            Buffer buffer = new Buffer();
            ByteBuffer resBuffer = ByteBuffer.wrap(msgBody);
            ResponseGetModel responseDataBuf = ResponseGetModel.getRootAsResponseGetModel(resBuffer);
            responseDataBuf.mutateIteration(curIter);
            buffer.write(responseDataBuf.getByteBuffer().array());
            return buffer;
        }
        return new Buffer();
    }

    private byte[] getMsgBodyFromFile(String resFileName) {
        byte[] res = null;
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(resFileName));
            res = (byte[]) ois.readObject();
            ois.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        return res;
    }

    public void run(int port) {
        server.setDispatcher(dispatcher);
        try {
            server.start(port);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    public void stop() {
        try {
            server.close();
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(0);
        }
    }
}
