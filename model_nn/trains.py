import torch
import os


def validation(model, testloader, criterion) -> tuple:
    """
    Function for measuring performance on test.

    :param model: nn to work with
    :param testloader: test data object
    :param criterion: criterion object
    :return: validation loss, accuracy
    """
    valid_loss = 0
    accuracy = 0

    # change model to work with cuda
    if torch.cuda.is_available():
        model.to('cuda')

    # Iterate over data from validloader
    for ii, (images, labels) in enumerate(testloader):
        # Change images and labels to work with cuda
        if torch.cuda.is_available():
            images, labels = images.to('cuda'), labels.to('cuda')

        # Forward pass image though model for prediction
        output = model.forward(images)
        # Calculate loss
        valid_loss += criterion(output, labels).item()
        # Calculate probability
        ps = torch.exp(output)

        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy


def train(model, trainloader, testloader, optimizer, criterion, lrscheduler, epoch: int = 15) -> torch.nn:
    """
    Training function

    :param model: nn to work with
    :param trainloader: train data object
    :param testloader: test data object
    :param optimizer: optimizer object
    :param criterion: criterion object
    :param lrscheduler: scheduler object
    :param epoch: num of epochs
    :return: trained nn
    """
    epochs = epoch
    steps = 0
    print_every = 40

    # change to gpu mode
    model.to('cuda')
    model.train()
    for e in range(epochs):

        running_loss = 0

        # Iterating over data to carry out training step
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            # zeroing parameter gradients
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Carrying out validation step
            if steps % print_every == 0:
                # setting model to evaluation mode during validation
                model.eval()

                # Gradients are turned off as no longer in training
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, testloader, criterion)

                print(f"No. epochs: {e + 1}, \
                Training Loss: {round(running_loss / print_every, 3)} \
                Valid Loss: {round(valid_loss / len(testloader), 3)} \
                Valid Accuracy: {round(float(accuracy / len(testloader)), 3)}")

                # Turning training back on
                model.train()
                lrscheduler.step(accuracy * 100)
    return model


def perf_measure(model, testloader) -> None:
    """
    Final performance measuring fn

    :param model: trained nn
    :param testloader: test data object
    :return: None
    """
    correct = 0
    total = 0
    if torch.cuda.is_available():
        model.to('cuda')

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.to('cuda'), labels.to('cuda')
            # Get probabilities
            outputs = model(images)
            # Turn probabilities into predictions
            _, predicted_outcome = torch.max(outputs.data, 1)
            # Total number of images
            total += labels.size(0)
            # Count number of cases in which predictions are correct
            correct += (predicted_outcome == labels).sum().item()

    print(f"Test accuracy of model: {round(100 * correct / total, 3)}%")


def find_classes(dir: str):
    """
    Function for getting labels

    :param dir: string of path
    :return: list of classes and dict of class:number
    """

    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx