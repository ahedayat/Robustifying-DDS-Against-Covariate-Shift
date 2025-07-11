import os 
import pickle 
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image as tensor2pil

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
        
        self.data_df = pd.read_csv(self.csv_path)



    def __getitem__(self, ix):
        image_path = self.data_df.loc[ix, "data_path"]
        label = self.data_df.loc[ix, "class_id"]

        image = Image.open(image_path)
        image = image.convert("RGB")
        image = ToTensor()(image)
        if image.shape[0] == 1:
            image = image.repeat([3, 1, 1]) 
        image = tensor2pil(image)


        image = self.transform(image)

        image = image.to(self.dtype)

        return image, label

    def __len__(self):
        return len(self.data_df)



