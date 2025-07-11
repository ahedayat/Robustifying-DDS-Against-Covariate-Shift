from utils import dict2namespace
from .diffusion_config import diffusion_config

unet_config = {
    "learn_sigma": diffusion_config.learn_sigma, #

    # Architecture
    "in_channels": 3, #OK
    "model_channels": 128, #OK
    "channel_mult": (1, 2, 2, 2), #OK
    "use_scale_shift_norm": True, #OK

    # residual blocks
    "num_res_blocks": 3, #OK
    # "resblock_updown": True, #

    # attention
    "attention_resolutions": (2, 4), #OK
    "num_heads": 4, #OK
    # "num_head_channels": 64, #
    "num_heads_upsample": -1, #OK
    # "use_new_attention_order": False,

    # Regularization
    "dropout": 0.3, #OK

    #fp-16
    # "use_fp16": True #

    # Pretrained model
    "pretrained_path": "./cifar10_uncond_50M_500K.pt", #OK

    "unet_input_size": (32, 32)
}

#here_unet_config

unet_config["out_channels"] = (
    unet_config["in_channels"] if not unet_config["learn_sigma"] 
    else 2*unet_config["in_channels"]
) #OK

unet_config = dict2namespace(unet_config)