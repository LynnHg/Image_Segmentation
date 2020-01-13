import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils_pkg import utils
from utils_pkg import config

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a.shape)
a = np.expand_dims(a, axis=2)
print(a.shape)
print(a)

