from torchvision import transforms


def train_transforms_prepare(size=(256, 256), normalize_mean=(0, 0, 0), normalize_std=(1, 1, 1),
                             center_crop=236, rotate=None, hflip=None, vflip=None, grayscale=None):
    """
    Preparing function for train data transformations

    :param size: size after resizing
    :param normalize_mean: mean for normalizing the inputs
    :param normalize_std: standard deviation for normalizing the inputs
    :param center_crop: size parameter, if int, then it will be square, if two-dim tuple, it will be rectangular
    :param rotate: number or a (min, max) tuple, in degrees, if number, then the rotation is on a
                   (-number, +number) range, uniformly, if (min, max), it is on (min, max) range
    :param hflip: probability of horizontal flip
    :param vflip: probability of vertical flip
    :param grayscale: probability of making the image grayscale (with 3 channels, r=g=b)
    :return: object of transformations
    """
    compose_list = [transforms.Resize(size)]

    if center_crop is not None:
        compose_list.append(transforms.CenterCrop(
            center_crop))

    if rotate is not None:
        compose_list.append(transforms.RandomRotation(
            rotate))

    if hflip is not None:
        compose_list.append(transforms.RandomHorizontalFlip(hflip))

    if vflip is not None:
        compose_list.append(transforms.RandomVerticalFlip(vflip))

    if grayscale is not None:
        compose_list.append(transforms.RandomGrayscale(
            grayscale))

    compose_list.append(transforms.ToTensor())
    compose_list.append(transforms.Normalize(normalize_mean, normalize_std))

    train_transforms = transforms.Compose(compose_list)
    return train_transforms


def test_transforms_prepare(size=(256, 256), center_crop=236, normalize_mean=(0, 0, 0), normalize_std=(1, 1, 1)):
    """
    Preparing function for test data transformations

    :param size: size after resizing
    :param center_crop: size parameter, if int, then it will be square, if two-dim tuple, it will be rectangular
    :param normalize_mean: mean for normalizing the inputs
    :param normalize_std: standard deviation for normalizing the inputs
    :return: object of transformations
    """
    test_transforms = transforms.Compose([transforms.Resize(size),
                                          transforms.CenterCrop(center_crop),
                                          transforms.ToTensor(),
                                          transforms.Normalize(normalize_mean, normalize_std)])
    return test_transforms
