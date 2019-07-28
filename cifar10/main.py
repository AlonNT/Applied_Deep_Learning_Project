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
from utils import get_data, train_model, get_accuracy, \
    get_accuracy_per_class, visualize_model, evaluate_model
import argparse
import datetime
from itertools import product
from ContinuousnessLoss import ContinuousnessLoss

# plt.ion()  # interactive mode


def main(net_type='ResNet20', lr=None, bs=None, momentum=None, epochs=5,
         use_checkpoint=False, verbose=False, embedding_continuousness_loss=False):
    # Set arguments with their default values. Since it's a list which is mutable,
    # it needs to be constructed this way and not in the function definition...
    if momentum is None:
        momentum = [0.9]
    if bs is None:
        bs = [32]
    if lr is None:
        lr = [0.1]

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

    checkpoint_path = './checkpoints/{}.pth'.format(net_type)
    if use_checkpoint and os.path.isfile(checkpoint_path):
        image_datasets, dataloaders, dataset_sizes, classes = get_data(bs=32, is_int=is_int)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device.type))

        # TODO export the following to a function to avoid code repetition

        evaluate_model(model, net_type, device, dataloaders, classes)

    else:
        for curr_lr, curr_bs, curr_momentum in product(lr, bs, momentum):
            print("Training with the following hyper-patameters:")
            print("learning-rate = {}".format(curr_lr))
            print("batch-size = {}".format(curr_bs))
            print("momentum = {}".format(curr_momentum))

            image_datasets, dataloaders, dataset_sizes, classes = get_data(curr_bs, is_int)

            # Define a Loss function and optimizer.
            # Let's use a Classification Cross-Entropy loss and SGD with momentum.
            criterion = nn.CrossEntropyLoss()
            embedding_loss = ContinuousnessLoss() if embedding_continuousness_loss else None

            optimizer = optim.SGD(model.parameters(), curr_lr, curr_momentum)

            # Decay LR by a factor of 0.1 every 1 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            # Train the network and save the best model
            model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                                dataloaders, dataset_sizes, device,
                                embedding_loss,
                                num_epochs=epochs, verbose=verbose)

            accuracy = evaluate_model(model, net_type, device, dataloaders, classes)

            # Will be changed to False if the current checkpoint accuracy is better.
            # Also, if the current execution did not train (only loaded a pre-trained
            # checkpoint) then no need to save)
            save_checkpoint = True

            # If there is already exists a checkpoint, check if we are better...
            if os.path.isfile(checkpoint_path):
                checkpoint_model = copy.deepcopy(model)
                checkpoint_model.load_state_dict(torch.load(checkpoint_path, map_location=device.type))
                checkpoint_acc = get_accuracy(dataloaders, device, checkpoint_model)

                if accuracy < checkpoint_acc:
                    save_checkpoint = False

            if save_checkpoint:
                torch.save(model.state_dict(), checkpoint_path)
                print("Checkpoint saved\nAccuracy = {:.2f}\nPrevious accuracy = {:.2f}".format(
                    accuracy, checkpoint_acc))

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
    parser.add_argument('--lr', default=[0.1], type=float, nargs='+',
                        help='learning-rate [default 0.1]')
    parser.add_argument('--bs', default=[32], type=int, nargs='+',
                        help='batch-size [default 32]')
    parser.add_argument('--momentum', default=[0.9], type=float, nargs='+',
                        help='momentum [default 0.9]')
    parser.add_argument('--epochs', default=5, type=int,
                        help='number of epochs [default 5]')
    parser.add_argument('--use_checkpoint', action='store_true',
                        help="If indicated and ./checkpoints/net_type.pth exists, use it")
    parser.add_argument('--verbose', action='store_true', 
                        help='progress bar and extra information')
    parser.add_argument('--embedding_continuousness_loss', action='store_true',
                        help='adds a loss for the embedding layer to be continuous')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.net_type, args.lr, args.bs, args.momentum, args.epochs,
         args.use_checkpoint, args.verbose, args.embedding_continuousness_loss)
