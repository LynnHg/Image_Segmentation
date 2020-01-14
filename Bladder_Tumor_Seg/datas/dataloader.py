from torch.utils.data import DataLoader

from utils_pkg import config
from datas.dataset import Dataset

def dataloader(rootpath=None, datatype="train", batch_size=1, shuffle=False):
    """
    :param rootpath: the root path of dataset
    :param datatype: available values: "train", "val", "test", default: "train"
    :return:
    """
    return DataLoader(dataset=Dataset(rootpath, datatype), batch_size=batch_size, shuffle=shuffle)
