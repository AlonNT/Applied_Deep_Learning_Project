import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

from tqdm import tqdm as tqdm
import datetime


class ToByteTensor(object):
    """
    Transform a PIL image to PyTorch ByteTensor,
    while permuting the dimensions H x W x C -> C x H x W
    This is needed in order to load original uint8 images (range 0-255)
    and not normalizing to float in range 0-1 (which is the torchvision default behaviour).
    """
    def __call__(self, picture):
        return torch.ByteTensor(np.array(picture)).permute(2, 0, 1)


class FromByteTensorToScaledFloat(object):
    """
    Transform a PyTorch ByteTensor image to PyTorch float tensor,
    while scaling it from  the range {0,1,...,255} to the range [0,1].
    """
    def __call__(self, picture):
        return picture.float().div(255)


class ToByteTensorShuffleColors(object):
    """
    Transform a PIL image to PyTorch ByteTensor,
    while permuting the dimensions H x W x C -> C x H x W.
    Afterwards, quantize each value (originally in the range of {0,1,...,255})
    to the range of {0,1,...,N}, and then multiply back to get values in the range
    {0,1,...,255}, but quantized (only N unique values, and not 256).
    """

    def __init__(self, embedding_size=32, seed=666):
        self.embedding_size = embedding_size
        self.seed = seed
        np.random.seed(seed)

        # Initialize as random permutation and then create the R,G,B coordinates
        quotient = np.random.permutation(embedding_size ** 3)

        quotient, r = np.divmod(quotient, embedding_size)  # R in the RGB representation
        quotient, g = np.divmod(quotient, embedding_size)  # G in the RGB representation
        b = np.mod(quotient, embedding_size)  # R in the RGB representation

        random_colors = np.column_stack((r, g, b)).astype(np.float32)
        random_colors = torch.tensor(random_colors)

        self.embeds = nn.Embedding(embedding_size ** 3, 3)
        self.embeds.weight.requires_grad = False
        self.embeds.load_state_dict({'weight': random_colors})

        self.factor = 256 / embedding_size
        self.monomials = torch.tensor([embedding_size ** 0, embedding_size ** 1, embedding_size ** 2])

    def __call__(self, picture):
        # Convert the PIL image to tensor
        x = torch.tensor(np.array(picture))

        # See the full documentation and explanation regarding this calculations in the models forward function
        # (e.g. SimpleConvNetWithEmbedding forward function).
        H, W, C = x.shape
        x /= self.factor
        x = x.view(H * W, 3)
        x = x.long()
        x = torch.matmul(x.float(), self.monomials.float()).long()
        x = x.view(H, W)
        x = self.embeds(x)
        x *= self.factor
        x = x.byte()
        x = x.permute(2, 0, 1)

        return x


def get_data(bs=32, is_int=False, shuffle_colors=False):
    """
    Get the CIFAR10 data.

    :param bs: the size of the minibatches to initialize the dataloaders
    :param is_int: whether to return the image as the original uint8,
                   or convert it to scaled float between 0 and 1.
    :param shuffle_colors: If indicated, shuffle the colors.
    :return: image_datasets: dictionary mapping "train"/"test" to its dataset
             dataloaders:    dictionary mapping "train"/"test" to its dataloader
             dataset_sizes:  dictionary mapping "train"/"test" to its dataset size
             classes:        tuple of 10 classes names in the correct order
    """
    if shuffle_colors:
        transforms = [ToByteTensorShuffleColors()]
    else:
        transforms = [ToByteTensor()]

    if not is_int:
        transforms = transforms + [FromByteTensorToScaledFloat()]

    transform = torchvision.transforms.Compose(transforms)

    image_datasets = {x: torchvision.datasets.CIFAR10(root='./data',
                                                      train=(x == 'train'),
                                                      download=True,
                                                      transform=transform)
                      for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=bs,
                                                  shuffle=True,
                                                  num_workers=2)
                   for x in ['train', 'test']}

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return image_datasets, dataloaders, dataset_sizes, classes


def train_model(model, criterion, optimizer, scheduler,
                dataloaders, dataset_sizes, device,
                embedding_loss=None, feed_embedding_loss_with='random',
                num_epochs=25, verbose=False):
    """
    A general function to train a model and return the best model found.

    :param model: the model to train
    :param criterion: which loss to train on
    :param optimizer: the optimizer to train with
    :param scheduler: scheduler for the learning rate
    :param dataloaders: the dataloaders to feed the model
    :param dataset_sizes: the sizes of the datasets
    :param device: which device to train on
    :param embedding_loss: whether or not to train with embedding-loss
    :param feed_embedding_loss_with: If 'images' - feed the embedding continuity loss with the images pixels,
                                     instead of random ones.
                                     If 'random' - choose random pixels to enforce continuity on.
    :param num_epochs: how many epochs to train
    :param verbose: whether or not to show progress bar during training

    :return: the model with the lowest test error
    """

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        tqdm.write('Epoch {}/{}'.format(epoch+1, num_epochs))
        tqdm.write('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_embeds_loss = 0.0
            running_criterion_loss = 0.0
            running_corrects = 0

            if verbose:
                pbar = tqdm(total=dataset_sizes[phase])

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()       # zero the parameter gradients

                # forward pass
                # track history if only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    running_criterion_loss += loss.item() * inputs.size(0)

                    # If there was given an embedding_loss, calculate it.
                    if embedding_loss is not None:
                        if feed_embedding_loss_with == 'random':
                            embeds_loss = embedding_loss(model.embeds)
                        elif feed_embedding_loss_with == 'images':
                            embeds_loss = embedding_loss(model.embeds, inputs)
                        else:
                            raise ValueError("feed_embedding_loss_with = {} " +
                                             "is not supported!".format(feed_embedding_loss_with))

                        loss += embeds_loss
                        running_embeds_loss += embeds_loss.item() * inputs.size(0)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if verbose:
                    pbar.update(dataloaders[phase].batch_size)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_embeds_loss = running_embeds_loss / dataset_sizes[phase]
            epoch_classification_loss = running_criterion_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            tqdm.write('{:5} Loss: {:.4f} Acc: {:.4f} Embedding-Loss: {:.4f} Classification-Loss: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_embeds_loss, epoch_classification_loss))

            # if the current model reached the best results so far,
            # deep copy the weights of the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    tqdm.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    tqdm.write('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, dataloaders, device, classes, num_images=6):
    """
    Visualizing the model predictions
    Generic function to display predictions for a few images

    :param model: the model to visualize
    :param dataloaders: the dataloaders to use
    :param device: which device to work on
    :param classes: tuple of the classes names
    :param num_images: how many images to show
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('Prediction:{} Truth: {}'.
                             format(classes[preds[j]],
                                    classes[labels.cpu().data[j]]))
                plt.imshow(np.transpose(inputs.cpu().data[j].numpy(), (1, 2, 0)))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def get_accuracy(dataloaders, device, model):
    """
    Get the model accuracy on the test data

    :param dataloaders: the dataloaders to use
    :param device: which device to work on
    :param model: the model to calculate accuracy

    :return: the accuracy (in percentages)
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def get_accuracy_per_class(dataloaders, device, model, classes):
    """
    Get the model accuracy on the test data per class

    :param dataloaders: the dataloaders to use
    :param device: which device to work on
    :param model: the model to calculate accuracy
    :param classes: tuple containing the classes names

    :return: a dictionary mapping class name to its accuracy (in percentages)
    """

    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    return {classes[i]: 100 * class_correct[i] / class_total[i]
            for i in range(len(classes))}


def evaluate_model(model, net_type, device, dataloaders, classes, visualize=False):
    """
    Evaluate a model - calculate its accuracy, accuracy per-class and visualize

    :param model: the model to evaluate
    :param net_type: the network type (used to save the figure of the visualizations)
    :param device: which device to work on
    :param dataloaders: the dataloaders to use
    :param classes: tuple containing the classes names
    :param visualize: whether or not to visualize samples

    :return: the accuracy of the model
    """
    accuracy = get_accuracy(dataloaders, device, model)

    tqdm.write('Accuracy of the network on the 10000 test images: {:.2f}'.format(accuracy))

    acc_per_cls = get_accuracy_per_class(dataloaders, device, model, classes)

    for class_name, class_accuracy in acc_per_cls.items():
        tqdm.write('Accuracy of {:5} : {:.2f}'.format(class_name, class_accuracy))

    if visualize:
        visualize_model(model, dataloaders, device, classes, num_images=6)
        curr_dt = datetime.datetime.now()
        fig_path = './logs/{}/{}_visual_samples.png'. \
            format(net_type, curr_dt.strftime("%Y-%m-%d_%H-%M-%S"))
        plt.savefig(fig_path)

    return accuracy
