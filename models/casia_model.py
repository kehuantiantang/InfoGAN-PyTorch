import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture based on InfoGAN paper.
"""


class Generator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc = nn.ConvTranspose2d(input_dim, 1024, 1, 1, bias=False)

        # self.fc = nn.Linear(233, 1024)

        self.tconv2 = nn.ConvTranspose2d(1024, 256, 8, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.tconv3 = nn.ConvTranspose2d(256, 256, 4, 1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.tconv4 = nn.ConvTranspose2d(256, 256, 4, 1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(256)

        self.tconv5 = nn.ConvTranspose2d(256, 128, 4, 2, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(128)

        self.tconv6 = nn.ConvTranspose2d(128, 64, 4, 2, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(64)


        self.tconv7 = nn.ConvTranspose2d(64, 3, 4, 1, padding=0, bias=False)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))
        x = F.relu(self.bn5(self.tconv5(x)))
        x = F.relu(self.bn6(self.tconv6(x)))
        x = self.tconv7(x)

        img = torch.tanh(x)

        return img


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 0)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 256, 4, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 256, 4, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1, inplace=True)
        # 2x2
        return x

class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        # fake or real
        self.conv = nn.Conv2d(256, 1, 2)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))
        return output


class QHead(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.conv1 = nn.Conv2d(256, 128, 2, bias=False)
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
        # ---------------table finish ------------------

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        # e
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var



if __name__ == '__main__':
    # g = Generator()
    # noise = torch.randn(128, 233, 1, 1)
    # r = g(noise)
    # print(r.size())

    d = Discriminator()
    netD = DHead()
    qhead = QHead()

    label = torch.randn(128, 3, 65, 65)
    dr = d(label)
    print(dr.size())

    nr = netD(dr)
    print(nr.size())


    qr = qhead(dr)
    for q in qr:
        print(q.size())

