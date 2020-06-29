import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import weights_init

"""
Architecture based on InfoGAN paper.
"""


class Generator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.tconv1 = nn.ConvTranspose2d(input_dim, 448, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(448)

        self.tconv2 = nn.ConvTranspose2d(448, 256, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)

        self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)

        self.tconv5 = nn.ConvTranspose2d(64, 3, 4, 2, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))

        img = torch.tanh(self.tconv5(x))

        return img


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)

        return x


class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(256, 1, 4)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))

        return output


class QHead(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.conv1 = nn.Conv2d(256, 128, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        dc = 1 if params.num_dis_c * params.dis_c_dim == 0 else params.num_dis_c * params.dis_c_dim

        # 10 categorical dimension
        self.conv_disc = nn.Conv2d(128, dc, 1)
        # two continuous codes

        cc = 1 if params.num_con_c == 0 else params.num_con_c

        self.conv_mu = nn.Conv2d(128, cc, 1)
        self.conv_var = nn.Conv2d(128, cc, 1)


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        # Not used during training for celeba dataset.
        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var



if __name__ == '__main__':
    pass
