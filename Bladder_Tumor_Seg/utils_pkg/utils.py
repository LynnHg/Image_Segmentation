import os
import random
import math
import cv2
import csv
import time, datetime
import numpy as np

import torch

from utils_pkg import config


def partition_data(dataset_dir):
    """
    Divide the raw data into training sets and validation sets
    :param dataset_dir: path root of dataset
    :return:
    """
    image_names = []
    label_names = []
    val_size = 0.2
    train_input_names = []
    train_label_names = []
    val_input_names = []
    val_label_names = []

    for file in os.listdir(os.path.join(dataset_dir, "Images")):
        cwd = os.getcwd()
        image_names.append(cwd + '/' + dataset_dir + 'Images/' + file)
        image_names.sort()
    for file in os.listdir(os.path.join(dataset_dir, "Labels")):
        cwd = os.getcwd()
        label_names.append(cwd + '/' + dataset_dir + 'Labels/' + file)
        label_names.sort()
    rawdata_size = len(image_names)
    random.seed(361)
    val_indices = random.sample(range(0, rawdata_size), math.floor(rawdata_size * val_size))
    train_indices = []
    for i in range(0, rawdata_size):
        if i not in val_indices:
            train_indices.append(i)
    with open(os.path.join(config.root_path, 'val.txt'), 'w') as f:
        for i in val_indices:
            val_input_names.append(image_names[i])
            f.write(image_names[i])
            f.write('\n')
    with open(os.path.join(config.root_path, 'val_labels.txt'), 'w') as f:
        for i in val_indices:
            val_label_names.append(label_names[i])
            f.write(label_names[i])
            f.write('\n')
    with open(os.path.join(config.root_path, 'train.txt'), 'w') as f:
        for i in train_indices:
            train_input_names.append(image_names[i])
            f.write(image_names[i])
            f.write('\n')
    with open(os.path.join(config.root_path,'train_labels.txt'), 'w') as f:
        for i in train_indices:
            train_label_names.append(label_names[i])
            f.write(label_names[i])
            f.write('\n')
    train_input_names.sort(), train_label_names.sort(), val_input_names.sort(), val_label_names.sort()
    return train_input_names, train_label_names, val_input_names, val_label_names


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image


def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1] - crop_width)
        y = random.randint(0, image.shape[0] - crop_height)

        if len(label.shape) == 3:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width, :]
        else:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width]
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (
            crop_height, crop_width, image.shape[0], image.shape[1]))


def data_augmentation(input_image, output_image):
    # Data augmentation
    input_image, output_image = random_crop(input_image, output_image, config.crop_height, config.crop_width)

    # if config.h_flip and random.randint(0, 1):
    #     input_image = cv2.flip(input_image, 1)
    #     output_image = cv2.flip(output_image, 1)
    # if config.v_flip and random.randint(0, 1):
    #     input_image = cv2.flip(input_image, 0)
    #     output_image = cv2.flip(output_image, 0)
    # if config.brightness:
    #     factor = 1.0 + random.uniform(-1.0 * config.brightness, config.brightness)
    #     table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    #     input_image = cv2.LUT(input_image, table)
    # if config.rotation:
    #     angle = random.uniform(-1 * config.rotation, config.rotation)
    # if config.rotation:
    #     M = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), angle, 1.0)
    #     input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]),
    #                                  flags=cv2.INTER_NEAREST)
    #     output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]),
    #                                   flags=cv2.INTER_NEAREST)

    return input_image, output_image


def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy

    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1])])
        # print(class_dict)
    return class_names, label_values


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

def log(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)

# 设置seed的函数，用于保证实验结果可再现
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True