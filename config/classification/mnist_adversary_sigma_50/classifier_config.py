from utils import dict2namespace
from torchvision.transforms import Normalize

classifier_config = {
    "name": "LeNet", # "beit", "LeNet"
    "model_path": "Karelito00/beit-base-patch16-224-pt22k-ft22k-finetuned-mnist",

    # "classifier_input_size": [224, 224],
    "classifier_input_size": [28, 28],

    # "classifier_normalizer": Normalize(
    #     [0.5, 0.5, 0.5],
    #     [0.5, 0.5, 0.5]
    # )
    "classifier_normalizer": Normalize(0.5, 0.5)
}

classifier_config = dict2namespace(classifier_config)