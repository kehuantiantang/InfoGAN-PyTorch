# InfoGAN-PyTorch

PyTorch implementation of [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) with result of experiments on *MNIST*, *Faces*, *CelebA* and *CASIA-Webface* datasets.

## Deployment Environment
* Ubuntu 16.04 LTS

* NVIDIA TITAN XP

* cuda 10.1

* Python 3.8.3,, recommend to install package using **conda**

  * `conda env create -f beta.yml`

  ```yaml
  name: beta
  channels:
    - pytorch
    - defaults
  dependencies:
    - _libgcc_mutex=0.1=main
    - blas=1.0=mkl
    - ca-certificates=2020.1.1=0
    - certifi=2020.6.20=py38_0
    - cudatoolkit=10.1.243=h6bb024c_0
    - freetype=2.10.2=h5ab3b9f_0
    - intel-openmp=2020.1=217
    - jpeg=9b=h024ee3a_2
    - ld_impl_linux-64=2.33.1=h53a641e_7
    - libedit=3.1.20191231=h7b6447c_0
    - libffi=3.3=he6710b0_1
    - libgcc-ng=9.1.0=hdf63c60_0
    - libgfortran-ng=7.3.0=hdf63c60_0
    - libpng=1.6.37=hbc83047_0
    - libstdcxx-ng=9.1.0=hdf63c60_0
    - libtiff=4.1.0=h2733197_1
    - lz4-c=1.9.2=he6710b0_0
    - mkl=2020.1=217
    - mkl-service=2.3.0=py38he904b0f_0
    - mkl_fft=1.1.0=py38h23d657b_0
    - mkl_random=1.1.1=py38h0573a6f_0
    - ncurses=6.2=he6710b0_1
    - ninja=1.9.0=py38hfd86e86_0
    - numpy=1.18.5=py38ha1c710e_0
    - numpy-base=1.18.5=py38hde5b4d6_0
    - olefile=0.46=py_0
    - openssl=1.1.1g=h7b6447c_0
    - pillow=7.1.2=py38hb39fc2d_0
    - pip=20.1.1=py38_1
    - python=3.8.3=hcff3b4d_0
    - pytorch=1.5.1=py3.8_cuda10.1.243_cudnn7.6.3_0
    - readline=8.0=h7b6447c_0
    - setuptools=47.3.1=py38_0
    - six=1.15.0=py_0
    - sqlite=3.32.3=h62c20be_0
    - tk=8.6.10=hbc83047_0
    - torchvision=0.6.1=py38_cu101
    - wheel=0.34.2=py38_0
    - xz=5.2.5=h7b6447c_0
    - zlib=1.2.11=h7b6447c_3
    - zstd=1.4.4=h0b5b093_3
    - pip:
      - absl-py==0.9.0
      - argparse==1.4.0
      - cachetools==4.1.0
      - chardet==3.0.4
      - easydict==1.9
      - google-auth==1.18.0
      - google-auth-oauthlib==0.4.1
      - grpcio==1.30.0
      - idna==2.9
      - markdown==3.2.2
      - oauthlib==3.1.0
      - protobuf==3.12.2
      - pyasn1==0.4.8
      - pyasn1-modules==0.2.8
      - pynvml==8.0.4
      - pyyaml==5.3.1
      - requests==2.24.0
      - requests-oauthlib==1.3.0
      - rsa==4.6
      - tensorboard==2.2.2
      - tensorboard-plugin-wit==1.6.0.post3
      - tqdm==4.46.1
      - urllib3==1.25.9
      - werkzeug==1.0.1
  prefix: /home/{username}/anaconda3/envs/beta
  ```

  

Edit the *argparse* in **`[train.py](train.py)`** file to select training parameters and the dataset to use.

```python
parser = argparse.ArgumentParser('InfoGAN')
# the training dataset, only can be ['celeba', 'faces', 'mnist', 'casia_webface'], you can refer the dataloader.py
parser.add_argument('--dataset', dest='dataset', help='Training dataset', default='faces', type=str)
parser.add_argument('--output_folder', dest='output_folder', help='the dir save result', default='output1', type=str)
parser.add_argument('--comment', dest='comment', help='comment', default='', type=str)
args = parser.parse_args()
```



Also, you can modify the {dataset}.yaml file to specify the training hyperparameters. This is a example of training hyperparamters:

```yaml
# Batch size.
batch_size: 126
# Number of epochs to train for.
num_epochs: 150
# Learning rate. adam beta1, beta2, specify for discriminator and generator 
D:
  lr: 0.0002
  beta1: 0.5
  beta2: 0.999
G:
  lr: 0.0002
  beta1: 0.5
  beta2: 0.999 
# After how many epochs to save checkpoints and generate test output.
save_epoch: 25
# Dataset name
dataset: 'MNIST'
# for repeatable
seed: 0

#lambda regularization term
coeff:
  # construction loss
  con: 0.1
  # discriminator
  dis: 1
# the range for continuous code in order to get fix noise
var:
  v1: -2
  v2: 2

# mnist 62 noise variables
num_z: 62
# c1, latent code(categorical code) --> model discontinuous variation
num_dis_c: 1
# categorical dimension, 10 classes, probability is 0.1 --> c1 ~ Cat(K = 10, p = 0.1)
dis_c_dim: 10
# c2, c3 continuous codes
num_con_c: 2
```
Example of training the dataset mnist
```sh
python3 train.py --dataset mnist --output_folder output 
```

## Results

* **MNIST** - [`mnist.md`](./README/mnist.md)
* **Faces** - [`faces.md`](./README/faces.md)
* **CelebA** - [`CelebA.md`](./README/CelebA.md)
* **CASIA_webface** - [`CASIA_webface.md`](./README/CASIA_webface.md)

* **Comparision result with β-VAE see  *README.md* in β-VAE**


## References
* [Natsu6767/InfoGAN-PyTorch](https://github.com/Natsu6767/InfoGAN-PyTorch)
