import torch
import torch.nn as nn
from itertools import product


class ContinuityLoss(nn.Module):

    def __init__(self, device, n_samples=1, p=2, embedding_size=256):
        super(ContinuityLoss, self).__init__()
        self.n_samples = n_samples
        self.p = p
        self.device = device
        self.embedding_size = embedding_size
        self.monomials = torch.tensor(data=[embedding_size ** 0, embedding_size ** 1, embedding_size ** 2],
                                      dtype=torch.int64, device=self.device)

        # There are 27 possible differences, for each diff in {-1,0,1}
        # it can be added to each of the R,G,B values
        self.diffs = torch.tensor(list(product(*[[-1, 0, 1]] * 3)),
                                  dtype=torch.int64, device=self.device).reshape(27, 1, 3)

        # This is the factor to divide each R,G,B value in order to quantize it correctly.
        # For example, if embedding_size is 32, the factor is 256/32 = 8.
        # So each value in (R,G,B) will be divided by 8 to get values between 0 and 32
        self.factor = 256 / embedding_size

    def forward(self, embeds, input_images=None):
        if input_images is None:
            # Generate n_samples random triplets of (R,G,B) values (in range 0,1,...,255)
            # A tensor of shape (n_samples, 3) where each row is a triplet representing
            # a specific (R,G,B) color to punish its distance from its neighbors.
            rgb_values = torch.randint(low=0, high=self.embedding_size, size=(self.n_samples, 3),
                                       dtype=torch.int64, device=self.device)
        else:
            # Sample n_samples random triplets of (R,G,B) values from the images RGB values.
            # A tensor of shape (n_samples, 3) where each row is a triplet representing
            # a specific (R,G,B) color to punish its distance from its neighbors.
            N, C, H, W = input_images.shape

            # Quantize the images colors wo be in range 0,1,...,embedding_size.
            input_images /= self.factor
            images_rgb_values = input_images.permute(0, 2, 3, 1).contiguous().view(N * H * W, C).long()
            permutation = torch.randperm(n=images_rgb_values.shape[0], device=self.device, dtype=torch.int64)
            rgb_values = images_rgb_values[permutation[:self.n_samples]]

        # Repeat the rgb_values 27 times to obtain a tensor of shape (27, n_samples, 3)
        # where each channel will be a neighbor of a specific kind
        # (i.e. (-1,-1,-1) or (1,-1,0) etc...)
        # Currently it's the same values 27 times but the differences will be added later.
        rgb_values_repeat = rgb_values.flatten().repeat(27).reshape(27, -1, 3)

        # Add the diffs to the rgb_values to obtain the neighbors.
        neighbors = rgb_values_repeat + self.diffs

        # clamp the values between 0 and 255 because the addition of 1 or -1 might
        # be out-of-bounds
        neighbors.clamp_(min=0, max=self.embedding_size - 1)

        # Multiplying rgb_values with the monomials (256^0, 256^1, 256^2)
        # gives n_samples indices (in the range [0,...,256^3 - 1])
        # and this is the index of the (R,G,B) color in the range [0,...,256^3 - 1]
        # in order to extract the proper embedding row.
        # Conversion to float is necessary to perform matmul in CUDA device.
        rgb_indices = torch.matmul(rgb_values.float(), self.monomials.float()).long()

        # Multiplying neighbors with the monomials (256^0, 256^1, 256^2)
        # gives a tensor of shape (27, n_samples) containing indices
        # (in the range [0,...,256^3 - 1]) which are the indices of the (R,G,B) colors
        # in order to extract the proper embedding rows.
        # The i-th row contain the n_samples neighbors all of a specific kind (e.g. -1,1,0)
        # Conversion to float is necessary to perform matmul in CUDA device.
        neighbors_indices = torch.matmul(neighbors.float(), self.monomials.float()).long()

        # Get the embedding of the n_samples RGB colors.
        # Will be a tensor of shape (n_samples, embedding_dim)
        # (embedding_dim = 3)
        rgb_embeddings = embeds(rgb_indices)

        # Get the embedding of the neighbors of the RGB colors.
        # Will be a tensor of shape (27, n_samples, embedding_dim)
        # (embedding_dim = 3)
        neighbors_embeddings = embeds(neighbors_indices)

        # rgb_embeddings is a tensor of shape (n_samples, embedding_dim) and contain
        # the embeddings of the randomly chosen RGB colors.
        # neighbors_embeddings a tensor of shape (27, n_samples, embedding_dim)
        # which contain the embeddings of their neighbors.
        # torch.dist will broadcast rgb_embeddings to be of shape
        # (27, n_samples, embedding_dim) and then calculate the p-norm
        # of the difference between the two tensors.
        distance = torch.dist(rgb_embeddings, neighbors_embeddings, self.p)

        return distance
