import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils import get_data
from create_embedding import create_rgb, create_random_poly


def get_embeddings(final_embedding_path, initial_embedding_path=None):
    """
    :param final_embedding_path: The path of the original embedding
    :param initial_embedding_path: The path of the reference embedding
    :return: Two nn Embedding layers, initialized with the given ones.
    """
    # Load the final embedding layer.
    final_embedding_matrix = np.load(final_embedding_path)
    final_embedding = nn.Embedding.from_pretrained(torch.tensor(final_embedding_matrix))

    # If initial_embedding_path was not given, initialize it as RGB.
    if initial_embedding_path is None:
        initial_embedding = create_rgb()

    # If it was a file with polynomial coefficients,
    # load them and create the corresponding embedding
    elif 'polynomial_coefficients' in initial_embedding_path:
        coefficients = np.load(initial_embedding_path)
        initial_embedding_matrix, _ = create_random_poly(coefficients)
        initial_embedding = nn.Embedding.from_pretrained(torch.tensor(initial_embedding_matrix))

    # Load the file as the actual embedding matrix.
    else:
        initial_embedding_matrix = np.load(initial_embedding_path)
        initial_embedding = nn.Embedding.from_pretrained(torch.tensor(initial_embedding_matrix))

    return final_embedding, initial_embedding


def visualize_embedding(final_embedding, initial_embedding, name="", num_images=12):
    """
    Visualize the two given embeddings, by showing the embeddings on num_images // 3
    random images taken from CIFAR-10 dataset.
    The original image (in RGB) is shown first, then the initial embedding,
    and finally the learned embedding.
    :param final_embedding: The final embedding layer
    :param initial_embedding: The initial embedding layer
    :param name: The of the PNG file to save
    :param num_images: Number of images to show in total
    """
    image_datasets, dataloaders, dataset_sizes, classes = get_data(bs=32, is_int=True)
    monomials = torch.tensor(data=[256 ** 0, 256 ** 1, 256 ** 2])

    # was_training = model.training
    images_so_far = 0
    plt.figure()

    with torch.no_grad():
        # See explanation about the following calculation in the
        # implementation of the embedding in the models
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            x = inputs

            N, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  # permute from N,C,H,W to N,H,W,C
            x = x.view(N, H * W, 3)
            x = x.long()
            x = torch.matmul(x.float(), monomials.float()).long()
            x = x.view(N, H, W)
            x_final_embedding = final_embedding(x)
            x_initial_embedding = initial_embedding(x)

            # permute from N,H,W,C to N,C,H,W
            outputs_main = x_final_embedding.permute(0, 3, 1, 2)
            outputs_ref = x_initial_embedding.permute(0, 3, 1, 2)

            for j in range(inputs.size()[0]):
                orig_im = np.transpose(inputs.data[j].numpy().astype(np.float32) / 255, (1, 2, 0))
                main_embed_im = np.transpose(outputs_main.data[j].numpy(), (1, 2, 0))
                ref_embed_im = np.transpose(outputs_ref.data[j].numpy(), (1, 2, 0))

                # Scale the images to be in range [0,1].
                # Each one of the three channels is being scaled independently of the others.
                main_embed_im = ((main_embed_im - main_embed_im.min(axis=(0, 1))) /
                                 (main_embed_im.max(axis=(0, 1)) - main_embed_im.min(axis=(0, 1))))
                ref_embed_im = ((ref_embed_im - ref_embed_im.min(axis=(0, 1))) /
                                (ref_embed_im.max(axis=(0, 1)) - ref_embed_im.min(axis=(0, 1))))

                images_so_far += 1
                ax = plt.subplot(num_images // 3, 3, images_so_far)
                plt.axis('off')
                ax.set_title("Original RGB")
                plt.imshow(orig_im)

                images_so_far += 1
                ax = plt.subplot(num_images // 3, 3, images_so_far)
                plt.axis('off')
                ax.set_title("Initial Embedding")
                plt.imshow(ref_embed_im)

                images_so_far += 1
                ax = plt.subplot(num_images // 3, 3, images_so_far)
                plt.axis('off')
                ax.set_title('After Embedding: dist {:.2f}'.format(np.linalg.norm(main_embed_im - ref_embed_im)))
                plt.imshow(main_embed_im)

                if images_so_far == num_images:
                    plt.savefig(os.path.join('.', 'embeddings', name + "_samples.png"))
                    plt.show()
                    return


def main(final_embedding_path, initial_embedding_path=None, visualize=False, name=""):
    """
    The main function.
    Arguments descriptions are given in the argparse help.
    """
    final_embedding, initial_embedding = get_embeddings(final_embedding_path, initial_embedding_path)

    distances = {p: torch.dist(final_embedding.weight, initial_embedding.weight, p).item()
                 for p in [0, 1, 2, np.inf]}

    for p, distance in distances.items():
        if p == 0:  # print the distance as an integer with commas
            print("The l_{:3} norm between the two embeddings is {:,}".format(p, int(distance)))
        else:  # print the distance as a float number
            print("The l_{:3} norm between the two embeddings is {:.2f}".format(p, distance))

    if visualize:
        visualize_embedding(final_embedding, initial_embedding, name)

    return distances


def parse_args():
    parser = argparse.ArgumentParser(description='Examine the learned embeddings - numerically and visually.')

    parser.add_argument('--final_embedding_path', required=True, type=str,
                        help="The path of the original embedding.")
    parser.add_argument('--initial_embedding_path', required=False, type=str,
                        help="The initial embedding path. "
                             "If initial_embedding_path is not given, initialize it as RGB."
                             "If it is a file with polynomial coefficients, "
                             "load them and create the corresponding embedding.")
    parser.add_argument('--visualize', action='store_true',
                        help="Whether to visualize the embeddings or not.")
    parser.add_argument('--name', type=str,
                        help="The name of the PNG file that will be saved.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.name is None:
        args.name = "{}_vs_{}".format(os.path.basename(args.final_embedding_path).replace('.npy', ''),
                                      os.path.basename(args.initial_embedding_path).replace('.npy', ''))

    main(args.final_embedding_path, args.initial_embedding_path, args.visualize, args.name)
