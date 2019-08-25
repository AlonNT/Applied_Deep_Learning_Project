import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os
import argparse
from time import strftime
from tqdm import tqdm
from itertools import product

from SimpleConvNet import SimpleConvNet, SimpleConvNetWithEmbedding
from resnet import resnet20, resnet20_with_embedding
from utils import get_data, train_model, evaluate_model
from create_embedding import create_rgb, create_random_poly
from ContinuityLoss import ContinuityLoss


def main(net_type='ResNet20',
         lr=None, bs=None, momentum=None, weight_decay=None,
         epochs=20, device_num=0,
         use_checkpoint=False, save_model=False, verbose=False,
         embedding_size=256, embedding_continuity_loss=-1,
         feed_embedding_loss_with='random', init_embedding_as='random'):
    """
    The main function.
    Arguments descriptions are given in the argparse help.
    """
    curr_time = strftime("%Y-%m-%d_%H-%M-%S")
    tqdm.write("Started at {}".format(curr_time))

    # If CUDA is available use the given device_num.
    # If not - use CPU.
    device = torch.device("cuda:{}".format(device_num)
                          if torch.cuda.is_available() else "cpu")
    tqdm.write("Using device: {}".format(device))

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

    # Initialize the embedding as the given option.
    if init_embedding_as == 'RGB':
        initial_embedding = torch.tensor(create_rgb(embedding_size), device=device)
        tqdm.write("Initializing the embedding layer with RGB.")

    # If the embedding is being initialized with a random polynomial,
    # save the coefficients to examine it later.
    elif init_embedding_as == 'random_poly':
        initial_embedding, coefficients = create_random_poly(embedding_size)
        initial_embedding = torch.tensor(initial_embedding, device=device)
        tqdm.write("Initializing the embedding layer with a random 3D polynomial.")
        if save_model:
            saved_coefficients_path = './embeddings/polynomial_coefficients_{}'.format(curr_time)
            tqdm.write("The coefficients are in the file {}".format(saved_coefficients_path))
            np.save(saved_coefficients_path, coefficients)

    # If the embedding is being initialized with random initialization,
    # save the embedding to examine it later.
    elif init_embedding_as == 'random':
        tqdm.write("Initializing the embedding layer with a random initialization " +
                   "(PyTorch default initialization).")
        initial_embedding = nn.Embedding(embedding_size**3, 3).weight.detach().cpu()
        if save_model:
            saved_embedding_path = './embeddings/random_embedding_{}'.format(curr_time)
            np.save(saved_embedding_path, initial_embedding)

    else:
        raise ValueError("init_embedding_as {} given is not supported!".format(init_embedding_as))

    # models with is_int=False receive float input scaled between 0 and 1
    # models with is_int=True receive original uint8 input (between 0 and 255)
    str_to_model = {'ResNet20': (resnet20, False),
                    'ResNet20WithEmbedding': (lambda: resnet20_with_embedding(device), True),
                    'SimpleConvNet': (SimpleConvNet, False),
                    'SimpleConvNetWithEmbedding': (lambda: SimpleConvNetWithEmbedding(device, embedding_size), True)}

    if net_type not in str_to_model:
        raise ValueError("net_type given is not a proper type\t{}".format(net_type))

    model_constructor, is_int = str_to_model[net_type]

    # This is the path of the checkpoint of this model.
    # Will be used to load the compare results to the current results, and to finally
    # save the results.
    checkpoint_path = './checkpoints/{}.pth'.format(net_type)
    embedding_path = './embeddings/{}'.format(net_type)

    # Check if the run mode is to use checkpoint and evaluate
    # (and thus the checkpoint file must exists).
    if use_checkpoint and os.path.isfile(checkpoint_path):
        # Build the model, weights are initialized randomly
        model = model_constructor().to(device)

        # Initialize the data-loaders to evaluate
        image_datasets, dataloaders, dataset_sizes, classes = get_data(bs=32, is_int=is_int)

        # Load the saved model from the checkpoint and evaluate the model
        checkpoint = torch.load(checkpoint_path, map_location=device.type)
        model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_state_dict)
        evaluate_model(model, net_type, device, dataloaders, classes)

        return

    # If save model is on, this training will save the model with the best accuracy,
    # regardless of whether the current saved model is better.
    if save_model:
        checkpoint_path = './checkpoints/{}_{}.pth'.format(net_type, curr_time)
        embedding_path = './embeddings/{}_{}'.format(net_type, curr_time)

    for curr_lr, curr_bs, curr_momentum, curr_weight_decay in \
            product(lr, bs, momentum, weight_decay):
        tqdm.write("#################################################################################")
        tqdm.write("#################################################################################")
        tqdm.write("#################################################################################")
        tqdm.write("Training with the following hyper-parameters:")
        tqdm.write("learning-rate = {}".format(curr_lr))
        tqdm.write("batch-size = {}".format(curr_bs))
        tqdm.write("momentum = {}".format(curr_momentum))
        tqdm.write("weight-decay = {}".format(curr_weight_decay))
        tqdm.write("#################################################################################")

        # Initialize the model
        model = model_constructor().to(device)

        # Initialize the embedding layer (if exists) with the initial embedding
        if 'embeds.weight' in model.state_dict().keys():
            model.embeds.load_state_dict({'weight': initial_embedding})

        # Initialize the data (data-loaders, classes names, etc...)
        image_datasets, dataloaders, dataset_sizes, classes = get_data(curr_bs, is_int)

        # Define a Loss function and optimizer.
        # We use a Classification Cross-Entropy loss,
        # and SGD with momentum and weight_decay.
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), curr_lr, curr_momentum, curr_weight_decay)

        # Decay LR by a factor of 0.5 every 10 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # Initialize the embedding continuity loss, if indicated.
        if embedding_continuity_loss != -1:
            embedding_loss = ContinuityLoss(device, n_samples=embedding_continuity_loss, embedding_size=embedding_size)
        else:
            embedding_loss = None

        # Train the network and return the best model (with the learned weights).
        model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                            dataloaders, dataset_sizes, device,
                            embedding_loss, feed_embedding_loss_with,
                            num_epochs=epochs, verbose=verbose)

        accuracy = evaluate_model(model, net_type, device, dataloaders, classes)

        if save_model:
            # Will be changed to False if the current checkpoint accuracy is better.
            # Also, if the current execution did not train (only loaded a pre-trained
            # checkpoint) then no need to save)
            model_good_enough_to_save = True
            checkpoint_acc = 0

            # If there is already exists a checkpoint, check if we are better...
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device.type)
                checkpoint_acc = checkpoint['accuracy']
                if accuracy < checkpoint_acc:
                    model_good_enough_to_save = False

            if model_good_enough_to_save:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'accuracy': accuracy,
                            'lr': curr_lr,
                            'bs': curr_bs,
                            'momentum': curr_momentum,
                            'weight_decay': curr_weight_decay},
                           checkpoint_path)
                # torch.save(model.state_dict(), checkpoint_path)
                tqdm.write("Checkpoint saved\nAccuracy = {:.2f}\nPrevious accuracy = {:.2f}".format(
                    accuracy, checkpoint_acc))
                if 'embeds.weight' in model.state_dict():
                    np.save(embedding_path, model.embeds.weight.detach().cpu().numpy())


def parse_args():
    parser = argparse.ArgumentParser(
        description='CIFAR-10 Training, with the ability to add embedding layer in the '
                    'beginning of the network, to enable learning the \"best\" color-space.'
    )

    parser.add_argument('--net_type', default='ResNet20', type=str,
                        help='Network type. ' +
                             'Must be one of the following: ' +
                             'ResNet20, ResNet20WithEmbedding, ' +
                             'ResNet20WithEmbedding, SimpleConvNetWithEmbedding. ' +
                             '[default is ResNet20]')

    parser.add_argument('--lr', default=[0.1], type=float, nargs='+',
                        help='[default 0.1]')
    parser.add_argument('--bs', default=[32], type=int, nargs='+',
                        help='[default 32]')
    parser.add_argument('--momentum', default=[0.9], type=float, nargs='+',
                        help='[default 0.9]')
    parser.add_argument('--weight_decay', default=[0], type=float, nargs='+',
                        help='[default 0, which means no weight-decay]')
    parser.add_argument('--epochs', default=5, type=int,
                        help='number of epochs [default 5].')

    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding matrix. [Default is 256]')
    parser.add_argument('--embedding_continuity_loss', type=int, default=-1,
                        help='How many samples to draw randomly to punish the ' +
                             'embedding-layer for being not-continuous. ' +
                             '[Default value is -1, which means no continuity_loss]')
    parser.add_argument('--feed_embedding_loss_with', type=str, default='images',
                        help='If \'images\' -  feed the embedding continuity loss with the '
                             'images\' pixels, instead of random ones. If \'random\' - choose random'
                             'pixels to enforce continuity on.'
                             '[Default is \'images\'')
    parser.add_argument('--init_embedding_as', type=str, default='random',
                        help='Initialize the embedding layer with some predefined embedding.')

    parser.add_argument('--device_num', type=int, default=0,
                        help='which device to train on ' +
                             '(will be used if CUDA is available)')
    parser.add_argument('--verbose', action='store_true',
                        help='progress bar and extra information')
    parser.add_argument('--use_checkpoint', action='store_true',
                        help="If indicated and ./checkpoints/net_type.pth exists, use it")
    parser.add_argument('--save_model', action='store_true',
                        help="If indicated, the model with the best test accuracy will be saved"
                             "(regardless of whether the existing saved model is better).")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    main(args.net_type, args.lr, args.bs, args.momentum, args.weight_decay, args.epochs,
         args.device_num, args.use_checkpoint, args.save_model, args.verbose,
         args.embedding_size, args.embedding_continuity_loss,
         args.feed_embedding_loss_with, args.init_embedding_as)
