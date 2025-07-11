import os 
import pickle 
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image as tensor2pil

from utils import (
    resize_tensor
)

class ImageNetLoader(Dataset):
    def __init__(
        self,
        csv_path,
        transform = None,
        dtype = torch.float32
    ):
        self.csv_path = csv_path
        self.transform = transform if transform is not None else ToTensor()        
        self.dtype = dtype

        self.common_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        self.data_df = pd.read_csv(self.csv_path)
        os.makedirs("./debug_images_imagenet/dataloader_before_transform", exist_ok=True)
        os.makedirs("./debug_images_imagenet/dataloader_after_transform", exist_ok=True)

    def normalize(self, X):
        # last_input_transform = self.transform.transforms[-1]
        # if isinstance(last_input_transform, torchvision.transforms.transforms.Normalize):
        #     return last_input_transform(X)
        return X


    def __getitem__(self, ix):
        image_path = self.data_df.loc[ix, "data_path"]
        label = self.data_df.loc[ix, "class_id"]

        image = Image.open(image_path)
        image = image.convert("RGB")
        image = ToTensor()(image)
        if image.shape[0] == 1:
            image = image.repeat([3, 1, 1]) 
        image = tensor2pil(image)

        # image.save(f"./debug_images_imagenet/dataloader_before_transform/{ix}.png")

        # image = self.transform(image)
        # image = ToTensor()(image)
        image = self.common_transform(image)


        # print(f">> Transform: {self.transform[-1]}")

        import torchvision
        # torchvision.transforms.functional.to_pil_image(image).save(f"./debug_images_imagenet/dataloader_after_transform/{ix}.png")

        image = image.to(self.dtype)

        return image, label

    def __len__(self):
        return len(self.data_df)



