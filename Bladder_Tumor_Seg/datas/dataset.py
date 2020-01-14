import os
import torch
import numpy as np
from utils_pkg import config, utils, helpers

# 获取分类类名和标签
class_names_list, label_values = helpers.get_label_info(os.path.join(config.root_path, "class_dict.csv"))

class Dataset:
    def __init__(self, rootpath: str, datatype: str = "train"):

        self.datatype = datatype
        # 加载数据集文件名
        train_input_names, \
        train_label_names, \
        val_input_names, \
        val_label_names = utils.partition_data(rootpath)

        if self.datatype == "train":
            self.input_names = train_input_names
            self.label_names = train_label_names
        elif self.datatype == "val":
            self.input_names = val_input_names
            self.label_names = val_label_names
        else:
            self.input_names = None
            self.label_names = None

    def __getitem__(self, index):
        input_image = utils.load_image(self.input_names[index])
        label_image = utils.load_image(self.label_names[index])
        # 增加通道维度，shape:[H,W] -> [H,W,C]
        input_image = np.expand_dims(input_image, axis=2)
        label_image = np.expand_dims(label_image, axis=2)
        input_image, label_image = utils.data_augmentation(input_image, label_image)
        # 归一化，one-hot编码，将每个像素点映射成类别向量
        input_image = np.float32(input_image) / 255.0
        label_image = np.float32(helpers.one_hot_it(label=label_image, label_values=label_values))
        # numpy to tensor，转换维度 shape:[H,W,C] -> [C,H,W]
        input_image = input_image.transpose([2, 0, 1])
        label_image = label_image.transpose([2, 0, 1])
        input_image = torch.from_numpy(input_image)
        label_image = torch.from_numpy(label_image)
        return input_image, label_image

    def __len__(self):
        return len(self.input_names)
