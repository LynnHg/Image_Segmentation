import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils_pkg import utils
from utils_pkg import config
import matplotlib.pyplot as plt
from datas import dataset, dataloader

# for i, j in dataloader.dataloader():
#     pass

colour_codes = np.array([[128], [255], [0]])
a = np.array([[0,0,0],[0,1,2],[2,2,2]])
print(a.shape)
x = colour_codes[a.astype(int)]

print(x)
print(x.shape)