import os 
import pickle 
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class ImageNet64Loader(Dataset):
    def __init__(
        self,
        data_root,
        transform = None,
        dtype = torch.float32
    ):
        self.w, self.h = 64, 64
        self.w_x_h = self.w * self.h
        
        self.data_root = data_root
        self.transform = transform if transform is not None else ToTensor()
        self.dtype = dtype
        
        self.batches = os.listdir(self.data_root)

        self.data = list()
        self.targets = list()

        for batch in tqdm(self.batches):
            batch_path = os.path.join(self.data_root, batch)
            with open(batch_path, "rb") as file:
                batch_pickle = pickle.load(file, encoding="latin1")

                batch_data = batch_pickle["data"]
                batch_targets = batch_pickle["labels"]

                batch_data = np.dstack(
                    (
                        batch_data[:, :self.w_x_h], 
                        batch_data[:, self.w_x_h:2*self.w_x_h], 
                        batch_data[:, 2*self.w_x_h:]
                    )
                )

                num_data_batch = batch_data.shape[0]
                batch_data = batch_data.reshape((num_data_batch, self.w, self.h, 3))

                self.data.append(batch_data)
                self.targets.extend(batch_targets)

        self.data = np.concatenate(self.data, axis=0)
        
                
    def __getitem__(self, ix):
        image = self.data[ix, :, :, :]
        target = self.targets[ix] - 1
        
        image = Image.fromarray(image)

        image = self.transform(image).to(self.dtype)

        return image, target

    def __len__(self):
        return self.data.shape[0]