from utils import dict2namespace

diffusion_config = {
    "pretrained_path": "1aurent/ddpm-mnist",
    "unet_input_size": (28, 28)
}

diffusion_config = dict2namespace(diffusion_config)