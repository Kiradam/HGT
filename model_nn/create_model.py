from torch import nn
from torch import optim
import torch
import torchvision.models as models


def create_model(model: nn, opti="sgd", lr=0.01, momentum=0.9, patience=3, threshold=0.9, *args, **kwargs) -> tuple:
    """
    Model defining function

    :param model: nn to work with
    :param opti: optimizer type
    :param lr: learning rate
    :param momentum: momentum of sgd
    :param patience: patience of scheduler
    :param threshold: threshold of scheduler
    :param args: *args
    :param kwargs: **kwargs
    :return: criterion, optimizer, scheduler objects
    """
    criterion = nn.CrossEntropyLoss()
    if opti == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif opti == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience, threshold=threshold)
    return criterion, optimizer, lrscheduler


def load_checkpoint(filepath: str):
    """
    Function for loading checkpoint

    :param filepath: string to checkpoint
    :return: loaded model
    """
    model = models.resnet34(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 48)
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()
    return model
