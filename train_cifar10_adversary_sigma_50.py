import os
import shutil

# from runners import TrainerRunner_v2
from runners import MultiGPUTrainerRunnerCifar10


def _main():
    print("train_cifar10.py")
    from config.classification.cifar10_adversary_sigma_50.data_config import data_config
    from config.classification.cifar10_adversary_sigma_50.classifier_config import classifier_config
    from config.classification.cifar10_adversary_sigma_50.data_config import data_config
    from config.classification.cifar10_adversary_sigma_50.diffusion_config import diffusion_config
    from config.classification.cifar10_adversary_sigma_50.train_config import train_config
    from config.classification.cifar10_adversary_sigma_50.unet_config import unet_config

    # shutil.copytree("./config/classification/cifar10_sigma_100", os.path.join(train_config.log.path, "configs"))

    cifar_runner = MultiGPUTrainerRunnerCifar10(
        data_config = data_config,
        diffusion_config = diffusion_config,
        unet_config = unet_config,
        classifier_config = classifier_config,
        train_config = train_config,
        device = train_config.device
    )

    cifar_runner.train()


if __name__ == "__main__":
    _main()