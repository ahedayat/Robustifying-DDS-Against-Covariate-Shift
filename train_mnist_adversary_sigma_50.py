import os
import shutil

from runners import MNISTTrainerRunner


def _main():
    print("train_cifar10.py")
    from config.classification.mnist_adversary_sigma_50.data_config import data_config
    from config.classification.mnist_adversary_sigma_50.classifier_config import classifier_config
    from config.classification.mnist_adversary_sigma_50.data_config import data_config
    from config.classification.mnist_adversary_sigma_50.diffusion_config import diffusion_config
    from config.classification.mnist_adversary_sigma_50.train_config import train_config

    mnist_runner = MNISTTrainerRunner(
        data_config = data_config,
        diffusion_config = diffusion_config,
        classifier_config = classifier_config,
        train_config = train_config,
        device = train_config.device
    )

    mnist_runner.train()


if __name__ == "__main__":
    _main()