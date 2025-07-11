import torch

from utils import dict2namespace

from .classifier_config import classifier_config
from .data_config import data_config

train_config = {
    # Data-Loader
    "data_loader": {
        "batch_size": 256,
        "num_workers": 1
    },

    # Criterion
    "criterion": {
        "type": "cross_entropy",
        "ce_coef": 1,
        "kl_coef": 100,
        "entropy_coef": 0.05
    },

    # Optimizer
    "optimizer": {
        "num_epochs": 100,
        "learning_rate": 1e-3,
        # "learning_rate": 1e-5,
        "weight_decay": 0.05,
        "gradient_accumulation_steps": 1,
    },

    "lr_scheduler":{
        "type": "cosine_annealing", # "cosine_annealing", "reduce_on_plateau"
        "eta_min": 1e-5
    },

    # diffusion
    "cat_diff":{
        "sigma": 1.0,
        "diffusion_end_step": None
    },

    # diffusion
    "adversary":{
        "optimizer": {
            "learning_rate": 1e-1,
            "weight_decay": 0.05,
        },
        "epsilon": torch.inf,
        "instance_adversary_iterations": 2,
        "adversary_perturbation": True
    },

    # Logging
    "log": {
        "freq": 10,
    },

    # "device": torch.device("cuda:1"),
    "device": torch.device("cuda"),

    # GPU resting
    "gpu_rest": {
        "freq": 5,
        "sec": 2 * 60,
    },

    "checkpoint": None
}

instance_adversary_iterations = train_config["adversary"]["instance_adversary_iterations"]
diffusion_sigma = train_config["cat_diff"]["sigma"]
train_config["log"]["path"] = f"/home/ahedayat/Documents/classifier_training/reports_new/{data_config.data_name}/{classifier_config.name}_adversarial_{instance_adversary_iterations}_sigma_{diffusion_sigma}" 

train_config = dict2namespace(train_config)