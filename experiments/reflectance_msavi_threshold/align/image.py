#!/usr/bin/env python
# coding: utf-8
"""
DJI P4 Image class

    An Image is a single file taken by a DJI P4 camera representing one
    band of multispectral information
"""


from pathlib import Path
from typing import Union

import cv2
import numpy as np
from skimage import io, transform, util
import matplotlib.pyplot as plt

import align.metadata as metadata

class Image(object):
    """
    Image class for DJI P4 Drone image. This is applicable to any image with a path."""
    def __init__(self, image_path: Union[str, Path], exiftool_obj=None):
        if not isinstance(image_path, Path):
            image_path = Path(image_path)
        if not image_path.is_file():
            raise IOError(f"Provided path is not a file: {image_path}")
        self.path = image_path
        # self.image = cv2.imread(str(self.path))
        self.meta = metadata.Metadata(self.path, exiftool_obj=exiftool_obj)
        self.height, self.width = self.gray.shape
        self.size = self.height, self.width

    def plot(self):
        plt.imshow(self.image)
        plt.show()