import torch
from torchvision.transforms import Resize, ToTensor, Compose

from utils import dict2namespace

data_config = {
    "data_name": "cifar10", 

    "num_classes": 10,

    # Image size
    "image_size": (32, 32),

    # Data path
    "train_path": "~/Documents/datasets/cifar10",
    "test_path": "~/Documents/datasets/cifar10",

    # Data Dtype
    "dtype": torch.float32,
    # "dtype": torch.float16,

    "num_workers": 2
}

data_config["input_transform"] = Compose([
    ToTensor()
])

data_config = dict2namespace(data_config)