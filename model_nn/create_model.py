<<<<<<< HEAD
version https://git-lfs.github.com/spec/v1
oid sha256:8155fdd0ad0938668731ec4c0aa00cbde9263887cc999f634c12815d9188f340
size 1530
=======
from torch import nn
from torch import optim


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
>>>>>>> affc212eb69e4ba073970100b267a6cd62802844
