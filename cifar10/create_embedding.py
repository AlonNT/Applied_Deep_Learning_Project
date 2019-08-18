import argparse
import numpy as np
from numpy.polynomial import polynomial as P
import os


def create_rgb():
    """
    Create an embedding containing the RGB.
    Each coordinate i in 0,...,256^3-1 will be the (R,G,B) of the same integers in
    basis of 256.
    The vectors are finally divided by 255 to be floats in [0,1].
    :return: the RGB embedding
    """
    # Get the R,G,B per index
    quotient = np.arange(256 ** 3)

    quotient, r = np.divmod(quotient, 256)   # R in the RGB representation
    quotient, g = np.divmod(quotient, 256)   # G in the RGB representation
    b = np.mod(quotient, 256)   # R in the RGB representation

    rgb_vectors = np.column_stack((r, g, b))

    rgb_vectors = rgb_vectors.astype(np.float32)
    rgb_vectors /= 255

    return rgb_vectors


def create_random_poly(coefficients=None, dim=3):
    """
    Create a polynomial embedding of the RGB vectors.
    If the coefficients are given then it is that polynomial, if not - some uniformly random
    polynomial is chosen, such that each variable is up to the given 'dim' degree
    :param coefficients: a 4D array, such that the 3D array coefficients[i] contains the
                         coefficients of the 3D polynomial of the i-th coordinate of
                         the 3D function (from (R,G,B) to another 3D space).
    :param dim: maximal power of the variables in the polynomial.
                i.e. if the dim is 3 than the monomial of the
                maximal degree will be x^3 * y^3 * z^3
    :return: polynomial mapping of the RGB vectors.
    """
    if coefficients is None:
        coefficients = np.random.uniform(low=-1, high=1, size=(3, dim, dim, dim))

    rgb_vectors = create_rgb()
    embedding = np.empty_like(rgb_vectors)

    for i in range(3):
        embedding[:, i] = P.polyval3d(rgb_vectors[:, 0], rgb_vectors[:, 1], rgb_vectors[:, 2],
                                      coefficients[i])

    return embedding, coefficients


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
