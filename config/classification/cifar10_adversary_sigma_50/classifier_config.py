from utils import dict2namespace

classifier_config = {
    "name": "resnet_v2",  # "resnet50", "resnet_v2", "swin_v2_b", "vit_cifar10"
    "model_path": "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
    "depth": 110,
    "classifier_input_size": (32, 32)
}

classifier_config = dict2namespace(classifier_config)