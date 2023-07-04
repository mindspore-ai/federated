# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""start set cross silo network"""

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor


class UNet(nn.Cell):
    """Unet"""
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16)

        self.upconv4 = nn.Conv2dTranspose(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.Conv2dTranspose(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.Conv2dTranspose(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.Conv2dTranspose(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def construct(self, x):
        """construct"""
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = ops.concat((dec4, enc4), axis=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = ops.concat((dec3, enc3), axis=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = ops.concat((dec2, enc2), axis=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = ops.concat((dec1, enc1), axis=1)
        dec1 = self.decoder1(dec1)
        return ops.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features):
        """define blocks"""
        return nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=features,
                    kernel_size=3,
                    padding=1,
                    has_bias=False,
                    pad_mode="pad",
                ),
                nn.BatchNorm2d(num_features=features),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=3,
                    padding=1,
                    has_bias=False,
                    pad_mode="pad",
                ),
                nn.BatchNorm2d(num_features=features),
                nn.ReLU(),
            ]
        )


def demo():
    import time
    model = UNet()
    x = Tensor(np.random.randn(10, 1, 256, 256).astype(np.float32))
    while True:
        start = time.time()
        y = model(x)
        end = time.time()
        print(end - start, y.shape)


if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU", device_id=7)
    demo()
