from torch.utils.data import DataLoader

from utils_pkg import config
from datas.dataset import Dataset

def dataloader():
    return DataLoader(dataset=Dataset(), batch_size=config.batch_size, shuffle=True)
