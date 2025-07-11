
import time
import datetime
import argparse

import torch
from torch.cuda.amp import autocast

class AverageMeter:
    """computes and stores the average and current value"""

    def __init__(
        self,
        start_val=0,
        start_count=0,
        start_avg=0,
        start_sum=0
    ):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
            Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
            Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = torch.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "num_corrects":
        return (top_ks == labels[:, None]).float().sum(dim=0).item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)

def min_max_scale(sample):
    min = torch.min(sample)
    max = torch.max(sample)

    return (sample - min) / (max - min)

def clamp(sample, min_value, max_value):
    return torch.clamp(sample, min_value, max_value)

def resize_tensor(X, size):
    with autocast():
        return torch.nn.functional.interpolate(
            X, 
            size, 
            mode='bicubic', 
            antialias=True
        )

def rest(sec):
    now = datetime.datetime.now()
    print(f"Don't Knock! I'm sleep for {sec} seconds. (@ {now})")

    time.sleep(sec)

    now = datetime.datetime.now()
    print(f"Hello, I woke up. (@ {now})")