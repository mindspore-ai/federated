package com.mindspore.flclient;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.security.SecureRandom;

import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.junit.Assert.*;

public class SecureProtocolTest {
    SecureProtocol secureProtocol;
    SecureRandom secureRandom;

    @Before
    public void setUp() throws Exception {
        secureProtocol = new SecureProtocol();
        secureRandom = new SecureRandom();
    }

    @After
    public void tearDown() throws Exception {
    }

    /**
     * Test whether the total output dimension of SignDS algorithm is 40 to 70, with the given parameters.
     */
    @Test
    public void signDSModel() {
        float signThrRatio = 0.6f;
        float signK = 0.2f;
        int inputDim = 800000;
        float signEps = 100f;
        secureProtocol.setDSParameter(signK, signEps, signThrRatio, 0);
        int topkDim = (int) (signK * inputDim);
        int signDimOut = secureProtocol.findOptOutputDim(signThrRatio, topkDim, inputDim, signEps);
        assertThat((double) signDimOut, closeTo(40, 70));
    }

    /**
     * Test whether the laplace noise is less than a certain magnitude with a certain probability under a given set of parameters.
     */
    @Test
    public void genLaplaceNoise() {
        int tryTime = 0;
        int allTryTime = 1600;
        float globalSensitivity = 1f;
        float eps = 230260f;
        float beta = globalSensitivity / eps;
        double threshold = 1e-4;
        double probability = 0.9f;
        for (int againTime = 0; againTime < allTryTime; againTime++) {
            if (secureProtocol.genLaplaceNoise(secureRandom, beta) < threshold) {
                tryTime++;
            }
        }
        assertThat((double) tryTime / allTryTime, greaterThan(probability));
    }
}