from torchvision.transforms import Normalize
from utils import dict2namespace

classifier_config = {
    # "name": "beitv3",  # "resnet50", "resnet_v2", "swin_v2_b", "vit_cifar10"beitv2
    "name": "beitv2",  # "resnet50", "resnet_v2", "swin_v2_b", "vit_cifar10"
    "model_path": "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
    "depth": 110,
    "classifier_input_size": (224, 224),
    # "classifier_input_size": (512, 512),
    # "classifier_normalizer": Normalize(
    #     [0.485, 0.456, 0.406],
    #     [0.229, 0.224, 0.225]
    # )
}

classifier_config = dict2namespace(classifier_config)