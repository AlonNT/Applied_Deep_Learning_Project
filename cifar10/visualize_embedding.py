from __future__ import print_function, division
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import get_data


def main(embedding_path):
    embedding = np.load(embedding_path)
    embedding_tensor = torch.tensor(embedding)
    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_tensor)
    image_datasets, dataloaders, dataset_sizes, classes = get_data(bs=32, is_int=True)
    num_images = 8
    monomials = torch.tensor(data=[256 ** 0, 256 ** 1, 256 ** 2])

    # was_training = model.training
    images_so_far = 0
    plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            x = inputs

            N, C, H, W = x.shape
            # Flatten the input tensor x to be in the shape of (batch, 32^2, 3)
            # in order to perform matrix multiplication with the vector
            # [255^0, 255^1, 255^2] and thus extracting the indices in the embedding
            # that correspond to each of the R,G,B triplet.
            x = x.permute(0, 2, 3, 1)  # permute from N,C,H,W to N,H,W,C
            x = x.view(N, H * W, 3)

            # Convert x from uint8 to long in order to multiply it with the
            # monomials 3D vector (which has type long).
            x = x.long()

            # Multiplying x with the monomials gives in each coordinate (b,i,j)
            # the multiplication of the i,j coordinate in the b-th image in the batch
            # with the vector (255^0, 255^1, 255^2) and this is the index of the
            # (R,G,B) color in the range [0,...,255^3 - 1] in order to extract the
            # proper embedding row.
            # Conversion to float is necessary to perform matmul in CUDA device.
            x = torch.matmul(x.float(), monomials.float()).long()

            # Reshape it back to (Batch, Width, Height), and now coordinate i,j is the
            # index of the (R,G,B) triplet in the embedding.
            x = x.view(N, H, W)

            # Now each coordinate i,j in x will be the 3D vector corresponding to the
            # R,G,B input triplet.
            x = embedding_layer(x)
            x = x.permute(0, 3, 1, 2)  # permute from N,H,W,C to N,C,H,W

            outputs = x

            for j in range(inputs.size()[0]):
                orig_im = np.transpose(inputs.data[j].numpy().astype(np.float32)/255, (1, 2, 0))
                embed_im = np.transpose(outputs.data[j].numpy(), (1, 2, 0))

                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                plt.axis('off')
                ax.set_title("dist {:.2f}".format(np.linalg.norm(orig_im - embed_im)))
                plt.imshow(np.transpose(inputs.data[j].numpy().astype(np.float32)/255, (1, 2, 0)))

                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                plt.axis('off')
                ax.set_title('After Embedding')
                plt.imshow(np.transpose(outputs.data[j].numpy(), (1, 2, 0)))

                if images_so_far == num_images:
                    plt.savefig(embedding_path + "_samples.png")
                    plt.show()
                    return


def parse_args():
    parser = argparse.ArgumentParser(description='Create embedding weights to initialize an embedding layer.')

    parser.add_argument('--embedding_path', required=True, type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.embedding_path)
