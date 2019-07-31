from __future__ import print_function, division
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import get_data, imshow


def main(embedding_path):
    embedding = np.load(embedding_path)
    embedding_tensor = torch.tensor(embedding)
    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_tensor)
    image_datasets, dataloaders, dataset_sizes, classes = get_data(bs=32, is_int=True)
    num_images = 8

    # was_training = model.training
    images_so_far = 0
    plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            outputs = embedding_layer(inputs)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.set_title('Original')
                imshow(inputs.data[j])

                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.set_title('After Embedding')
                imshow(outputs.data[j])

                if images_so_far == num_images:
                    return


def parse_args():
    parser = argparse.ArgumentParser(description='Create embedding weights to initialize an embedding layer.')

    parser.add_argument('--embedding_path', required=True, type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.embedding_path)
