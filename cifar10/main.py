from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os
import copy
from SimpleConvNet import SimpleConvNet, SimpleConvNetWithEmbedding
from resnet import resnet20, resnet20_with_embedding
from utils import get_data, train_model, get_accuracy, evaluate_model
import argparse
from tqdm import tqdm
from itertools import product
from ContinuousnessLoss import ContinuousnessLoss


def main(net_type='ResNet20', lr=None, bs=None, momentum=None, weight_decay=None,
         epochs=20, device_num=0,
         use_checkpoint=False, verbose=False,
         embedding_continuousness_loss=False, embedding_loss_n_samples=1,
         init_embedding_as_rgb=False):
    """
    The main function. arguments description are given in the argparse help.
    """
    # Set arguments with their default values. Since it's a list which is mutable,
    # it needs to be constructed this way and not in the function definition...
    if bs is None:
        bs = [32]
    if lr is None:
        lr = [0.1]
    if momentum is None:
        momentum = [0.9]
    if weight_decay is None:
        weight_decay = [0.001]

    device = torch.device("cuda:{}".format(device_num)
                          if torch.cuda.is_available() else "cpu")
    tqdm.write("Using device: {}".format(device))

    # models with is_int=False receive float input scaled between 0 and 1
    # models with is_int=True receive original uint8 input (between 0 and 255)
    str_to_model = {'ResNet20': (resnet20, False),
                    'ResNet20WithEmbedding': (lambda: resnet20_with_embedding(device), True),
                    'SimpleConvNet': (SimpleConvNet, False),
                    'SimpleConvNetWithEmbedding': (lambda: SimpleConvNetWithEmbedding(device), True)}

    if net_type not in str_to_model:
        raise ValueError("net_type given is not a proper type\t{}".format(net_type))

    model_constructor = str_to_model[net_type][0]
    is_int = str_to_model[net_type][1]

    # This is the path of the checkpoint of this model.
    # Will be used to load the compare results to the current results, and to finally
    # save the results.
    checkpoint_path = './checkpoints/{}.pth'.format(net_type)

    # Check if the run mode is to use checkpoint and evaluate
    # (and thus the checkpoint file must exists).
    if use_checkpoint and os.path.isfile(checkpoint_path):
        # Build the model, weights are initialized randomly
        model = model_constructor().to(device)

        # Initialize the data-loaders to evaluate
        image_datasets, dataloaders, dataset_sizes, classes = get_data(bs=32, is_int=is_int)

        # Load the saved model from the checkpoint and evaluate the model
        model.load_state_dict(torch.load(checkpoint_path, map_location=device.type))
        evaluate_model(model, net_type, device, dataloaders, classes)

    # This mode is training mode, and afterwards the results will be saved if they
    # are better than the checkpoint results.
    else:
        if init_embedding_as_rgb:
            rgb_embedding = np.load("./embeddings/RGB.npy")
            rgb_embedding_tensor = torch.tensor(rgb_embedding, device=device)

        for curr_lr, curr_bs, curr_momentum, curr_weight_decay in product(lr, bs, momentum, weight_decay):
            tqdm.write("################################################################")
            tqdm.write("Training with the following hyper-parameters:")
            tqdm.write("learning-rate = {}".format(curr_lr))
            tqdm.write("batch-size = {}".format(curr_bs))
            tqdm.write("momentum = {}".format(curr_momentum))
            tqdm.write("weight-decay = {}".format(curr_weight_decay))

            # Initialize the model
            model = model_constructor().to(device)
            if init_embedding_as_rgb:
                if 'embeds.weight' not in model.state_dict().keys():
                    raise ValueError("init_embedding_as_rgb flag was given, but the " +
                                     "model does not include an Embedding layer!")

                model.embeds.load_state_dict({'weight': rgb_embedding_tensor})

            # Initialize the data (data-loaders, classes names, etc...)
            image_datasets, dataloaders, dataset_sizes, classes = get_data(curr_bs, is_int)

            # Define a Loss function and optimizer.
            # We use a Classification Cross-Entropy loss,
            # and SGD with momentum and weight_decay.
            criterion = nn.CrossEntropyLoss()

            if embedding_continuousness_loss:
                embedding_loss = ContinuousnessLoss(device,
                                                    n_samples=embedding_loss_n_samples)
            else:
                embedding_loss = None

            optimizer = optim.SGD(model.parameters(), curr_lr, curr_momentum, curr_weight_decay)

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
            checkpoint_acc = 0

            # If there is already exists a checkpoint, check if we are better...
            if os.path.isfile(checkpoint_path):
                checkpoint_model = copy.deepcopy(model)
                checkpoint_model.load_state_dict(torch.load(checkpoint_path, map_location=device.type))
                checkpoint_acc = get_accuracy(dataloaders, device, checkpoint_model)

                if accuracy < checkpoint_acc:
                    save_checkpoint = False

            if save_checkpoint:
                torch.save(model.state_dict(), checkpoint_path)
                tqdm.write("Checkpoint saved\nAccuracy = {:.2f}\nPrevious accuracy = {:.2f}".format(
                    accuracy, checkpoint_acc))
                if 'embeds.weight' in model.state_dict():
                    np.save('./embeddings/{}_embedding'.format(net_type),
                            model.embeds.weight.detach().cpu().numpy())


def parse_args():
    parser = argparse.ArgumentParser(
        description='CIFAR-10 Training, with the ability to add embedding layer in the '
                    'beginning of the network, to enable learning the \"best\" color-space.'
    )

    parser.add_argument('--net_type', default='ResNet20', type=str)
    parser.add_argument('--lr', default=[0.1], type=float, nargs='+',
                        help='learning-rate [default 0.1]')
    parser.add_argument('--bs', default=[32], type=int, nargs='+',
                        help='batch-size [default 32]')
    parser.add_argument('--momentum', default=[0.9], type=float, nargs='+',
                        help='momentum [default 0.9]')
    parser.add_argument('--weight_decay', default=[0], type=float, nargs='+',
                        help='momentum [default 0, which means no weight-decay]')
    parser.add_argument('--epochs', default=20, type=int,
                        help='number of epochs [default 5]')
    parser.add_argument('--use_checkpoint', action='store_true',
                        help="If indicated and ./checkpoints/net_type.pth exists, use it")
    parser.add_argument('--verbose', action='store_true', 
                        help='progress bar and extra information')
    parser.add_argument('--embedding_continuousness_loss', action='store_true',
                        help='adds a loss for the embedding layer to be continuous')
    parser.add_argument('--embedding_loss_n_samples', type=int, default=1,
                        help='How many samples to draw randomly to punish the ' +
                             'embedding-layer for being not-continuous')
    parser.add_argument('--init_embedding_as_rgb', action='store_true',
                        help='Initialize the embedding layer to be RGB ' +
                             '(i.e. the identity mapping)')
    parser.add_argument('--device_num', type=int, default=0,
                        help='which device to train on ' +
                             '(will be used if CUDA is available)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.net_type, args.lr, args.bs, args.momentum, args.weight_decay, args.epochs,
         args.device_num, args.use_checkpoint, args.verbose,
         args.embedding_continuousness_loss, args.embedding_loss_n_samples,
         args.init_embedding_as_rgb)
