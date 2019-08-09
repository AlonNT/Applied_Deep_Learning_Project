import torch
import torch.nn as nn
import random
from itertools import product


class ContinuousnessLoss(nn.Module):

    def __init__(self, device, n_samples=1, p=2):
        super(ContinuousnessLoss, self).__init__()
        self.n_samples = n_samples
        self.p = p
        self.device = device

    def forward(self, embeds):
        loss = 0

        num_embeddings, embedding_dim = embeds.num_embeddings, embeds.embedding_dim

        # Draw n_samples of colors (i.e. triplets of (R,G,B) values).
        # For each color sampled, calculate the difference between its embedding vector
        # and its neighbors' embeddings
        for i in range(self.n_samples):
            # Generate a random index in the embedding layer, corresponds to a (R,G,B color)
            rgb_idx = random.randint(0, num_embeddings - 1)

            # Calculate the R,G,B values from the index
            b = rgb_idx % 256  # B in the RGB representation
            quotient = rgb_idx // 256
            g = quotient % 256  # G in the RGB representation
            quotient = quotient // 256
            r = quotient % 256  # R in the RGB representation

            assert r * 256 ** 2 + \
                   g * 256 ** 1 + \
                   b * 256 ** 0 == rgb_idx, \
                "Bad Calculation of RGB from index"

            indices = list()

            for d_r, d_g, d_b in product(*[[-1, 0, 1]] * 3):
                bad_idx = False
                # prevent overflow - if any of the R,G,B coordinate are out-of-bounds
                # mark this idx as invalid to prevent adding it to the indices list
                for value, d_value in [(r, d_r), (g, d_g), (b, d_b)]:
                    if value + d_value < 0 or value + d_value > 255:
                        bad_idx = True

                if not bad_idx:
                    idx = (r + d_r) * 256 ** 2 + \
                          (g + d_g) * 256 ** 1 + \
                          (b + d_b) * 256 ** 0
                    indices += [idx]

            rgb_idx_tensor = torch.tensor([rgb_idx]).long().to(self.device)
            indices = torch.tensor(indices).long().to(self.device)

            color_embedding = embeds(rgb_idx_tensor)
            neighbors_embeddings = embeds(indices)
            distance = torch.dist(color_embedding, neighbors_embeddings, p=2)

            loss += distance

        return loss
