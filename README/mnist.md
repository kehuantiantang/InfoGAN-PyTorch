## Train InfoGAN on MNIST dataset

### 1. Configuration

Refer to the [config](../config/mnist.yaml) file, and [model](../models/mnist_model.py),  different than **C1.MNIST**, I use Convolution layer to replace FC in discriminator network, and ConvTransposed2d to replace FC in generator network.



### 2. Training Curve

![Discriminator & Generator Loss](./res/mnist/loss1.png)

![Total Loss](./res/mnist/loss2.png)



### 3. Manipulating Latent codes on MNIST

<table align='center'>
<tr align='center'>
<th> Random Generator </th>
<th> Varing c1 on InfoGAN </th>
</tr>
<tr>
<td><img src = 'res/mnist/random.png' height = '450'>
<td><img src = 'res/mnist/c1.png' height = '450'>
</tr>
<th> Varing c2 from -2 to 2 on InfoGAN </th>
<th> Varing c3 from -2 to 2 on InfoGAN </th>  
<tr>
<td><img src = 'res/mnist/c2.png' height = '450'>
<td><img src = 'res/mnist/c3.png' height = '450'>
</tr>
</table>



Training change

<table align='center'>
<tr align='center'>
<th> Random </th>
<th> c3 </th>
<th> c2 </th>
</tr>
<tr>
<td><img src = 'res/mnist/fixed.gif' height = '300'>
<td><img src = 'res/mnist/dcs00_ccs00.gif' height = '300'>
<td><img src = 'res/mnist/dcs00_ccs01.gif' height = '300'>
</tr>
</table>

