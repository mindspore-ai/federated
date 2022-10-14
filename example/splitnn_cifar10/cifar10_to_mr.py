# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Vertically split and convert Cifar10 images to mindrecord."""

from importlib import import_module
from mindspore import log as logger
from mindspore.mindrecord import Cifar10ToMR
from mindspore.mindrecord import FileWriter
from mindspore.mindrecord import SUCCESS
from mindspore.mindrecord import FAILED
from mindspore.mindrecord.tools.cifar10 import Cifar10

try:
    cv2 = import_module("cv2")
except ModuleNotFoundError:
    cv2 = None


__all__ = ['SplitCifar10ToMR']


class SplitCifar10ToMR(Cifar10ToMR):
    """
    A class to split and transform from cifar10 to MindRecord.

    Args:
        source (str): The cifar10 directory to be transformed.
        destination (str): MindRecord file path to transform into, ensure that no file with the same name
            exists in the directory.

    Raises:
        ValueError: If source or destination is invalid.
    """
    def run(self, fields=None):
        """
        Split and transform cifar10 to MindRecord.

        Args:
            fields (list[str], optional): A list of index fields. Default: None. For index field settings,
                please refer to :func:`mindspore.mindrecord.FileWriter.add_index`.

        Returns:
            MSRStatus, SUCCESS or FAILED.
        """
        if fields and not isinstance(fields, list):
            raise ValueError("The parameter fields should be None or list")

        # Load training and test images/labels
        cifar10_data = Cifar10(self.source, False)
        cifar10_data.load_data()
        train_images, train_labels = cifar10_data.images, cifar10_data.labels
        test_images, test_labels = cifar10_data.Test.images, cifar10_data.Test.labels

        # Split the images vertically
        train_images_left, train_images_right = train_images[:, :, :16, :], train_images[:, :, 16:, :]
        test_images_left, test_images_right = test_images[:, :, :16, :], test_images[:, :, 16:, :]

        # Construct raw images and convert them to mindrecord
        train_left = _construct_raw_data(train_images_left, train_labels)
        train_right = _construct_raw_data(train_images_right, train_labels)
        test_left = _construct_raw_data(test_images_left, test_labels)
        test_right = _construct_raw_data(test_images_right, test_labels)

        if _generate_mindrecord(self.destination + "_train_left", train_left, fields, "train_left") != SUCCESS:
            return FAILED
        if _generate_mindrecord(self.destination + "_train_right", train_right, fields, "train_right") != SUCCESS:
            return FAILED
        if _generate_mindrecord(self.destination + "_test_left", test_left, fields, "test_left") != SUCCESS:
            return FAILED
        if _generate_mindrecord(self.destination + "_test_right", test_right, fields, "test_right") != SUCCESS:
            return FAILED
        return SUCCESS

def _construct_raw_data(images, labels):
    """
    Construct raw data from cifar10 data.

    Args:
        images (list): image list from cifar10.
        labels (list): label list from cifar10.

    Returns:
        list[dict], data dictionary constructed from cifar10.
    """

    if not cv2:
        raise ModuleNotFoundError("opencv-python module not found, please use pip install it.")

    raw_data = []
    for i, img in enumerate(images):
        label = int(labels[i][0])
        _, img = cv2.imencode(".jpeg", img[..., [2, 1, 0]])
        row_data = {"id": int(i),
                    "data": img.tobytes(),
                    "label": int(label)}
        raw_data.append(row_data)
    return raw_data


def _generate_mindrecord(file_name, raw_data, fields, schema_desc):
    """
    Generate MindRecord file from raw data.

    Args:
        file_name (str): File name of MindRecord File.
        fields (list[str]): Fields would be set as index which
          could not belong to blob fields and type could not be 'array' or 'bytes'.
        raw_data (dict): dict of raw data.
        schema_desc (str): String of schema description.

    Returns:
        MSRStatus, SUCCESS or FAILED.
    """

    schema = {"id": {"type": "int64"}, "label": {"type": "int64"},
              "data": {"type": "bytes"}}

    logger.info("transformed MindRecord schema is: {}".format(schema))

    writer = FileWriter(file_name, 1, overwrite=True)
    writer.add_schema(schema, schema_desc)
    if fields and isinstance(fields, list):
        writer.add_index(fields)
    writer.write_raw_data(raw_data)
    return writer.commit()
