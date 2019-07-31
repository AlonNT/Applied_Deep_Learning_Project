from __future__ import print_function, division
import argparse
import numpy as np
import os


def main(embedding_type, output_dir):
    # Get the R,G,B per index
    quotient = np.arange(256 ** 3)

    quotient, r = np.divmod(quotient, 256)   # R in the RGB representation
    quotient, g = np.divmod(quotient, 256)   # G in the RGB representation
    b = np.mod(quotient, 256)   # R in the RGB representation

    rgb_vectors = np.column_stack((r, g, b))

    rgb_vectors = rgb_vectors.astype(np.float32)
    rgb_vectors /= 255

    np.save(os.path.join(output_dir, embedding_type), rgb_vectors)


def parse_args():
    parser = argparse.ArgumentParser(description='Create embedding weights to initialize an embedding layer.')

    parser.add_argument('--embedding_type', default='RGB', type=str)
    parser.add_argument('--output_dir', default='./embeddings', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.embedding_type, args.output_dir)
