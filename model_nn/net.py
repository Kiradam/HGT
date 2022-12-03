from torch import nn
from torchvision import models


def net(pre_type: str = "resnet", num_label: int = 48) -> nn:
    """
    Net preparing function

    :param pre_type: pre-built nn type
    :param num_label: number of labels
    :return: extended neural network object
    """
    model = models.resnet34(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_label)
    if pre_type == "densenet":
        model = models.densenet121(weights='DEFAULT')
        model.classifier = nn.Linear(num_ftrs, num_label)

    return model
