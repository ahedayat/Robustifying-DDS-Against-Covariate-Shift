import torch
from torchvision.transforms import Resize, ToTensor, Compose

from utils import dict2namespace

data_config = {
    "data_name": "imagenet", 

    "num_classes": 1000,

    # Image size
    "image_size": (224, 224),

    # Data path
    # "train_path": "~/Documents/datasets/cifar10",
    # "test_path": "~/Documents/datasets/cifar10",
    "train_path": "./imagenet_train_data.csv",
    "test_path": "./imagenet_val_data.csv",

    # Data Dtype
    "dtype": torch.float32,
    # "dtype": torch.float16,

    "num_workers": 2
}

data_config["input_transform"] = Compose([
    ToTensor()
])

data_config = dict2namespace(data_config)