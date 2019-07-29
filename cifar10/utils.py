import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from tqdm import tqdm as tqdm
import datetime


class ToTensorWithoutScaling(object):
    """H x W x C -> C x H x W"""
    def __call__(self, picture):
        return torch.ByteTensor(np.array(picture)).permute(2, 0, 1)


def get_data(bs=32, is_int=False):
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Check if need to return the image as the original uint8
    # or convert it to scaled float between 0 and 1 (and then normalize).
    if is_int:
        # transform = transforms.Compose([np.array, torch.tensor])
        transform = ToTensorWithoutScaling()
    else:
        # The output of torchvision datasets are PILImage images of range [0, 1].
        # We transform them to Tensors of normalized range [-1, 1].
        transform = transforms.ToTensor()
        # transform = transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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


def imshow(img):
    """
    Show an image after un-normalizing it
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def train_model(model, criterion, optimizer, scheduler,
                dataloaders, dataset_sizes, device,
                embedding_loss=None,
                num_epochs=25, verbose=False):
    """
    A general function to train a model and return the best model found.
    """

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_embeds_loss = 0.0
            running_classification_loss = 0.0
            running_corrects = 0

            if verbose:
                pbar = tqdm(total=dataset_sizes[phase])

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    running_classification_loss += loss.item() * inputs.size(0)

                    # If there was given an embedding_loss, calculate it.
                    if embedding_loss is not None:
                        embeds_loss = embedding_loss(model.embeds)
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
            epoch_classification_loss = running_classification_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{:5} Loss: {:.4f} Acc: {:.4f} Embedding-Loss: {:.4f} Classification-Loss: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_embeds_loss, epoch_classification_loss))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, dataloaders, device, classes, num_images=6):
    """
    Visualizing the model predictions
    Generic function to display predictions for a few images
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
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def get_accuracy(dataloaders, device, model):
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
    # what are the classes that performed well,
    # and the classes that did not perform well:

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


def evaluate_model(model, net_type, device, dataloaders, classes):
    accuracy = get_accuracy(dataloaders, device, model)

    print('Accuracy of the network on the 10000 test images: {:.2f}'.format(accuracy))

    acc_per_cls = get_accuracy_per_class(dataloaders, device, model, classes)

    for class_name, class_accuracy in acc_per_cls.items():
        print('Accuracy of {:5} : {:.2f}'.format(class_name, class_accuracy))

    # visualize_model(model, dataloaders, device, classes, num_images=6)
    # curr_dt = datetime.datetime.now()
    # fig_path = './logs/{}/{}_visual_samples.png'. \
    #     format(net_type, curr_dt.strftime("%Y-%m-%d_%H-%M-%S"))
    # plt.savefig(fig_path)

    return accuracy
