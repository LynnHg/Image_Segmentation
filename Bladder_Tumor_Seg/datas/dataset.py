
from utils_pkg import config, utils

class Dataset:
    def __init__(
            self, rootpath: str=config.root_path,
            datatype: str="train"
    ):

        self.datatype = datatype
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
        print(input_image.shape)

    def __len__(self):
        return len(self.input_names)
