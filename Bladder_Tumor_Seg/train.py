import os
import torch
import random
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from utils_pkg import utils, helpers
from datas import dataloader
from utils_pkg import config
from models_pkg.Unet import Unet

np.set_printoptions(threshold=9999999)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn([3, 1, 160, 160])
x = x.to(device)
model = Unet(1, 2)
model = model.to(device)
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.7)

for epoch in range(1, config.num_epoches + 1):
    current_losses = []
    count = 0
    # 开始时间
    st = time.time()
    epoch_st = time.time()

    # 开始训练
    for inputs, labels in dataloader.dataloader(config.root_path, "train", config.batch_size, True):
        print(inputs.shape)
        X = inputs.to(device)
        y = labels.to(device)
        optimizer.zero_grad()
        output = model(X)
        output = torch.sigmoid(output)
        loss = criterion(output, y)
        loss.backward()
        current = loss.item()
        current_losses.append(current)
        optimizer.step()
        count = count + config.batch_size
        if count % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f" % (
                epoch, count, current, time.time() - st)
            utils.log(string_print)
            st = time.time()
