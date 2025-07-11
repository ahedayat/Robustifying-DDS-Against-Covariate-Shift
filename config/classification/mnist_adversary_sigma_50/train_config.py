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
        "num_epochs": 20,
        "learning_rate": 5e-05,
        "weight_decay": 0.05,
        "gradient_accumulation_steps": 1,
    },

    "lr_scheduler":{
        "type": "cosine_annealing", # "cosine_annealing", "reduce_on_plateau"
        "eta_min": 1e-5
    },

    # diffusion
    "sigma": 0.50,

    # diffusion
    "adversary":{
        "optimizer": {
            "learning_rate": 1e-1,
            "weight_decay": 0.05,
        },
        # "epsilon": torch.inf,
        "epsilon": 1e-1,
        "instance_adversary_iterations": 1,
        "adversary_perturbation": True
        # "adversary_perturbation": False
    },

    # Logging
    "log": {
        "freq": 10,
    },

    "device": torch.device("cuda:0"),
    # "device": torch.device("cuda"),

    "checkpoint": None
}

instance_adversary_iterations = train_config["adversary"]["instance_adversary_iterations"]
epsilon = train_config["adversary"]["epsilon"]
diffusion_sigma = train_config["sigma"] if train_config["sigma"] != torch.inf else "inf" 
train_config["log"]["path"] = f"/home/ahedayat/Documents/classifier_training/reports_new/{data_config.data_name}/{classifier_config.name}_adversarial_{instance_adversary_iterations}_{epsilon}_sigma_{diffusion_sigma}" 

train_config = dict2namespace(train_config)