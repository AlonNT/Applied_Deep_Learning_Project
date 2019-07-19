from __future__ import print_function, division

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
# import torchvision
# from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
# import time
import os
import copy
# from tqdm import tqdm as tqdm
from SimpleConvNet import SimpleConvNet, SimpleConvNetWithEmbedding
from resnet import resnet20, resnet20_with_embedding
from utils import get_data, train_model, get_accuracy, get_accuracy_per_class, visualize_model
import argparse
import random

# plt.ion()  # interactive mode


def main(net_type='ResNet20', learning_rate=0.1, momentum=0.9, num_epochs=5,
         verbose=False, use_checkpoint=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    if net_type == 'ResNet20':
        model = resnet20().to(device)
        is_int = False  # This model receive float input scaled between 0 and 1
    elif net_type == 'ResNet20WithEmbedding':
        model = resnet20_with_embedding().to(device)
        is_int = True  # This model receive float input scaled between 0 and 1
    elif net_type == 'SimpleConvNet':
        model = SimpleConvNet().to(device)
        is_int = False  # This model receive float input scaled between 0 and 1
    elif net_type == 'SimpleConvNetWithEmbedding':
        model = SimpleConvNetWithEmbedding().to(device)
        is_int = True  # This model receive original uint8 input scaled between 0 and 255
    else:
        raise ValueError("net_type given was is not a proper type\t{}".format(net_type))

    image_datasets, dataloaders, dataset_sizes, classes = get_data(is_int)

    checkpoint_path = './checkpoints/{}.pth'.format(net_type)
    if use_checkpoint and os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device.type))
    else:
        # Define a Loss function and optimizer.
        # Let's use a Classification Cross-Entropy loss and SGD with momentum.
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        # Decay LR by a factor of 0.1 every 1 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # Train the network and save the best model
        model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                            dataloaders, dataset_sizes, device,
                            num_epochs=num_epochs, verbose=verbose)

    accuracy = get_accuracy(dataloaders, device, model)

    print('Accuracy of the network on the 10000 test images: {:.2f}'.format(accuracy))

    acc_per_cls = get_accuracy_per_class(dataloaders, device, model, classes)

    for class_name, class_accuracy in acc_per_cls.items():
        print('Accuracy of {:5} : {:.2f}'.format(class_name, class_accuracy))

    #visualize_model(model, dataloaders, device, classes, num_images=6)

    #plt.show()

    # Will be changed to False if the current checkpoint accuracy is better.
    save_checkpoint = True
    if os.path.isfile(checkpoint_path):
        checkpoint_model = copy.deepcopy(model)
        checkpoint_model.load_state_dict(torch.load(checkpoint_path, map_location=device.type))
        checkpoint_acc = get_accuracy(dataloaders, device, checkpoint_model)

        if accuracy < checkpoint_acc:
            save_checkpoint = False

    if save_checkpoint:
        torch.save(model.state_dict(), checkpoint_path)

    # # If Embedding layer is in the model,
    # # save it as numpy binary file and as a text file.
    # if 'embeds.weight' in model.state_dict():
    #     np.save('./checkpoints/{}_embedding'.format(net_type),
    #             model.embeds.weight.detach().cpu().numpy())
    #     np.savetxt('./checkpoints/{}_embedding.txt'.format(net_type),
    #                model.embeds.weight.detach().cpu().numpy())


def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')

    parser.add_argument('--net_type', default='ResNet20', type=str)
    parser.add_argument('--use_checkpoint', action='store_true',
                        help="If indicated and ./checkpoints/net_type.pth exists, use it")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--num_epochs', default=5, type=int, help='number of epochs')
    parser.add_argument('--verbose', action='store_true', help='progress bar and extra information')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.net_type, args.lr, args.momentum, args.num_epochs, args.verbose, args.use_checkpoint)
