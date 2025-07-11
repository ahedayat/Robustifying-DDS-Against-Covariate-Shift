from utils import dict2namespace

diffusion_config = {
    # Architecture
    "diffusion_steps" : 1000, #OK
    "predict_xstart": False,
    
    # Σ
    "learn_sigma": True, #OK
    "sigma_small": False, #

    # Noise
    "noise_schedule": "linear", #OK
    "noise_schedule_sampler": "uniform", # Noise Scheduler Sampler

    # Loss
    "use_kl": False, #OK

    # Others
    "predict_xstart": False, #OK
    "rescale_timesteps": False, #OK
    "timestep_respacing": "250", #OK
    "rescale_learned_sigmas": False, #OK
}

diffusion_config = dict2namespace(diffusion_config)