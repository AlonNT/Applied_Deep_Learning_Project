import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    """
    A simple Convolutional Neural Network.
    The architecture is:
    Conv-Relu-Pool-Conv-Relu-Pool-Affn-Relu-Affn-Relu-Affn
    """

    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleConvNetWithEmbedding(nn.Module):
    """
    A simple Convolutional Neural Network.
    The architecture is:
    Conv-Relu-Pool-Conv-Relu-Pool-Affn-Relu-Affn-Relu-Affn
    """

    def __init__(self, device, embedding_size=256):
        super(SimpleConvNetWithEmbedding, self).__init__()

        self.embeds = nn.Embedding(embedding_size**3, 3)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.device = device
        self.embedding_size = embedding_size

        # Multiplying the 3D vector (x,y,z) by the monomials vector will give the index in the embedding matrix.
        self.monomials = torch.tensor(data=[embedding_size**0, embedding_size**1, embedding_size**2],
                                      device=self.device)

        # This is the factor to divide each R,G,B value in order to quantize it correctly.
        # For example, if embedding_size is 32, the factor is 256/32 = 8.
        # So each value in (R,G,B) will be divided by 8 to get values between 0 and 32
        self.factor = 256 / embedding_size

    def forward(self, x):
        N, C, H, W = x.shape

        # Quantize the images colors wo be in range 0,1,...,embedding_size.\
        x /= self.factor

        # Flatten the input tensor x to be in the shape of (batch, 32^2, 3)
        # in order to perform matrix multiplication with the vector
        # [255^0, 255^1, 255^2] and thus extracting the indices in the embedding
        # that correspond to each of the R,G,B triplet.
        x = x.permute(0, 2, 3, 1)   # permute from N,C,H,W to N,H,W,C
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
        x = torch.matmul(x.float(), self.monomials.float()).long()

        # Reshape it back to (Batch, Width, Height), and now coordinate i,j is the
        # index of the (R,G,B) triplet in the embedding.
        x = x.view(N, H, W)

        # Now each coordinate i,j in x will be the 3D vector corresponding to the
        # R,G,B input triplet.
        x = self.embeds(x)
        x = x.permute(0, 3, 1, 2)  # permute from N,H,W,C to N,C,H,W

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
