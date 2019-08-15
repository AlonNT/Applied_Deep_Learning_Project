from __future__ import print_function, division
import argparse
import numpy as np
import os


def create_rgb():
    # Get the R,G,B per index
    quotient = np.arange(256 ** 3)

    quotient, r = np.divmod(quotient, 256)   # R in the RGB representation
    quotient, g = np.divmod(quotient, 256)   # G in the RGB representation
    b = np.mod(quotient, 256)   # R in the RGB representation

    rgb_vectors = np.column_stack((r, g, b))

    rgb_vectors = rgb_vectors.astype(np.float32)
    rgb_vectors /= 255

    return rgb_vectors


def create_random_poly():
    max_degree = 10
    n_coefficients = 10

    rgb_vectors = create_rgb()

    # # powers contain the powers for the three variables of the polynomial,
    # # for each one of the n_coefficients (which are also chosen randomly).
    # powers = np.random.randint(low=0, high=max_degree+1, size=(n_coefficients, 3))
    # coefficients = np.random.uniform(low=0, high=1, size=n_coefficients)

    embedding = np.empty_like(rgb_vectors)

    for i in range(3):
        coefficients = np.random.uniform(low=-1, high=1, size=(3, 3, 3))
        embedding[:, i] = np.polynomial.polynomial.polyval3d(rgb_vectors[:, 0], rgb_vectors[:, 1], rgb_vectors[:, 2],
                                                             coefficients)

    return embedding


def main(embedding_type, output_dir):
    if embedding_type == "RGB":
        embedding_func = create_rgb
    elif embedding_type == "random_poly":
        embedding_func = create_random_poly
    else:
        raise ValueError("embedding_type given is not supported!")

    embedding = embedding_func()
    np.save(os.path.join(output_dir, embedding_type), embedding)


def parse_args():
    parser = argparse.ArgumentParser(description='Create embedding weights to initialize an embedding layer.')

    parser.add_argument('--embedding_type', default='RGB', type=str)
    parser.add_argument('--output_dir', default='./embeddings', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.embedding_type, args.output_dir)
