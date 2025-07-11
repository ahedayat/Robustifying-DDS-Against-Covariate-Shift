from utils import dict2namespace

diffusion_config = {
    # Architecture
    "diffusion_steps" : 4000, #OK
    
    # Σ
    "learn_sigma": True, #OK
    "sigma_small": False, #OK

    # Noise
    "noise_schedule": "cosine", #OK
    "noise_schedule_sampler": "uniform", # Noise Scheduler Sampler

    # Loss
    "use_kl": False, #OK

    # Others
    "predict_xstart": False, #OK
    "rescale_timesteps": True, #OK
    "timestep_respacing": "", #OK
    "rescale_learned_sigmas": True, #OK
}

diffusion_config = dict2namespace(diffusion_config)