import pprint
import yaml
from easydict import EasyDict as edict

# dataset = 'celeba'
def get_params(dataset):
    if dataset == 'mnist':
        f = './config/mnist.yaml'
    elif dataset == 'svhn':
        yaml.load('./config/svhn.yaml')
    elif dataset == 'celeba':
        f = './config/celeba.yaml'
        # params = yaml.load('./config/celeba.yaml')
    elif dataset == 'faces':
        f = './config/faces.yaml'
        # params = yaml.load('./config/faces.yaml')
    elif dataset == 'casia_webface' or dataset.lower() == 'casia':
        f = './config/casia_webface.yaml'
        # params = yaml.load('./config/casia_webface.yaml')
    else:
        raise ValueError('%s is not exist'%dataset)

    print('Read file %s'%f)
    file =  open(f, 'r', encoding='utf-8').read()
    params = yaml.load(file,  Loader=yaml.FullLoader)

    pprint.pprint(params)
    params = edict(params)
    return params





if __name__ == '__main__':
    get_params('celeba')