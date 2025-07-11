import os 
import pickle 
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from transformers.models.vit.image_processing_vit import ViTImageProcessor

class TinyImageNetLoader(Dataset):
    def __init__(
        self,
        csv_path,
        transform = None,
        dtype = torch.float32
    ):
        self.csv_path = csv_path
        self.dtype = dtype
        self.transform = transform if transform is not None else ToTensor()

        self.data_df = pd.read_csv(csv_path)

    def __getitem__(self, ix):
        _, image_path, class_name, class_id = self.data_df.iloc[ix,:]

        image = Image.open(image_path)
        image = image.convert("RGB")

        if isinstance(self.transform, ViTImageProcessor):
            image = self.transform(image, return_tensors="pt")
            image = image["pixel_values"][0,:,:,:]
        else: 
            image = self.transform(image)

        if image.shape[0] == 1:
            image = image.repeat([3, 1, 1])

        image = image.to(self.dtype)

        return image, class_id

    def __len__(self):
        return len(self.data_df)
