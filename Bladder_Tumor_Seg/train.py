import os
import torch
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils_pkg import utils
from utils_pkg import config
from models_pkg.Unet import Unet
# np.set_printoptions(threshold=9999999)

train_input_names, train_label_names, val_input_names, val_label_names = utils.partition_data(config.root_path)
class_names_list, label_values = utils.get_label_info(os.path.join(config.root_path, "class_dict.csv"))
num_classes = len(label_values)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(1, 2)
model = model.to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.7)

for epoch in range(1, config.num_epoches + 1):
    train_loss  = []

    # Equivalent to shuffling
    id_list = np.random.permutation(len(train_input_names))

    num_iters = int(np.floor(len(id_list) / config.batch_size))
    for i in range(num_iters):
        input_image_batch = []
        label_image_batch = []

        # Collect a batch of images
        for j in range(config.batch_size):
            index = i * config.batch_size + j
            id = id_list[index]
            input_image = utils.load_image(train_input_names[id])
            label_image = utils.load_image(train_label_names[id])

            input_image = np.expand_dims(input_image, axis=2)
            label_image = np.expand_dims(label_image, axis=2)

            input_image, label_image = utils.data_augmentation(input_image, label_image)

            # Prep the data. Make sure the labels are in one-hot format
            input_image = np.float32(input_image) / 255.0
            label_image = np.float32(utils.one_hot_it(label=label_image, label_values=label_values))


            input_image_batch.append(np.expand_dims(input_image, axis=0))
            label_image_batch.append(np.expand_dims(label_image, axis=0))


        if config.batch_size == 1:
            input_image_batch = input_image_batch[0]
            label_image_batch = label_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            label_image_batch = np.squeeze(np.stack(label_image_batch, axis=1))

        input_image_batch = np.expand_dims(input_image_batch, axis=3)
        input_image_batch = input_image_batch.transpose([0, 3, 1, 2])
        label_image_batch = label_image_batch.transpose([0, 3, 1, 2])
        input_image_batch = torch.from_numpy(input_image_batch)
        label_image_batch = torch.from_numpy(label_image_batch)

        X = input_image_batch.to(device)
        y = label_image_batch.to(device)
        optimizer.zero_grad()
        output = model(X)
        output = torch.sigmoid(output)
        loss = criterion(output, y)
        loss.backward()
        iter_loss = loss.item()
        train_loss += iter_loss
        optimizer.step()

        print(train_loss)
