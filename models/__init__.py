# coding=utf-8
from utils import weights_init


def prepare_model(params, device):
    if (params['dataset'] == 'MNIST'):
        from models.mnist_model import Generator, Discriminator, DHead, QHead
    elif (params['dataset'] == 'SVHN'):
        from models.svhn_model import Generator, Discriminator, DHead, QHead
    elif (params['dataset'] == 'CelebA'):
        from models.celeba_model import Generator, Discriminator, DHead, QHead
    elif (params['dataset'] == 'FashionMNIST'):
        from models.mnist_model import Generator, Discriminator, DHead, QHead
    elif (params['dataset'] == 'Faces'):
        from models.faces_model import Generator, Discriminator, DHead, QHead
    elif (params['dataset'] == 'Casia'):
        from models.casia_model import Generator, Discriminator, DHead, QHead
    else:
        raise ValueError()


    input_dim = params.num_z + params.num_dis_c * params.dis_c_dim + params.num_con_c

    netG = Generator(input_dim).to(device)
    netG.apply(weights_init)
    print(netG)

    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)
    print(discriminator)

    netD = DHead().to(device)
    netD.apply(weights_init)
    print(netD)

    netQ = QHead(params).to(device)
    netQ.apply(weights_init)
    print(netQ)


    return netG, netQ, netD, discriminator


