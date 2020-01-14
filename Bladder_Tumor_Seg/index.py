import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils_pkg import utils
from utils_pkg import config
import matplotlib.pyplot as plt
from datas import dataset, dataloader

for images, labels in dataloader.dataloader(config.root_path, "val"):
    print(images.shape)
    print(labels.shape)

