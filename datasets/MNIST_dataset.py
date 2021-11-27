import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from datasets.base_dataset import BaseDataset
from utils.augmenters.augment import seg
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import struct

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

class MNISTDataset(BaseDataset):
    """
    Input params:
        stage: The stage of training.
        configuration: Configuration dictionary.
    """
    def __init__(self, configuration):
        super().__init__(configuration)

        self._stage = configuration["stage"]

        self._image_size = tuple(configuration["input_size"])

        self.dataset_path = os.path.join(configuration["dataset_path"])

        #-----------------------------------------------------------------------
        #Here is where you can do things like preload data and labels or do image preprocessing
        self.images = []
        self.labels = []

        if (self._stage == 'train'):
            images = read_idx(os.path.join(self.dataset_path, 'train-images.idx3-ubyte'))
            labels = read_idx(os.path.join(self.dataset_path, 'train-labels.idx1-ubyte'))
        elif (self._stage == 'val'):
            images = read_idx(os.path.join(self.dataset_path, 't10k-images.idx3-ubyte'))
            labels = read_idx(os.path.join(self.dataset_path, 't10k-labels.idx1-ubyte'))
        self.images = images
        self.labels = np.int_(labels)
        #-----------------------------------------------------------------------


        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    #This function returns an data, label pair. All data processing and modification should be done by the end of this function
    def __getitem__(self, index):
        #Load data from xml, you will need to modify if annotations are in different format
        image = self.images[index]

        #Image loading assuming the images are in the 'images' folder in the dataset root path
        image = np.asarray(image)#.reshape(48, 48)
        image = image.astype(np.uint8)

        #Image resizing
        image = cv2.resize(image, self._image_size)

        #Image formatting
        image = np.dstack([image] * 1)

        #Some image augmentation
        image = seg(image=image)

        #Apply defined transforms to image from constructor (will convert to tensor)
        image = self._transform(image)
        #Ensure the target is set as the label
        target = self.labels[index]

        #image should be the image data, target should be the label
        return image, target

    def __len__(self):
        # return the size of the dataset, replace with len of labels array
        return len(self.labels)
