# Autoencoders

This module is a set of several models of autoencoders with a handy constructor. This constructor takes the autoencoder configuration as yaml file and produce complete pytorch model ready to train. It also can be a base for custom implementation of the autoencoder logic.

## Installation

The are default deep learning packages that are listed in requirenments.txt, but there is one package named `hypytorch` that should be installed manually or by `setup.py`: 

#### manually:
1. Clone git repository https://github.com/leymir/hyperbolic-image-embeddings
2. `cd hyperbolic-image-embeddings` 
3. `python setup.py install` 
#### by `setup.py` (`hyptorch` will be installed like manually) 
```bash
pip install .
```


## About config

On the high level, the config represents the autoencoder with all parameters, the big parts of which are encoder and decoder structure. Let's have a look at the example

```yaml
model_cfg: 
  arch : ae
  encoder :
      arch : fc_vec
      linear_layer_type: euclidian
      l_hidden: 
      - 64
      activation : 
      - relu
      out_activation: linear
  decoder : 
      arch : fc_vec
      linear_layer_type: euclidian
      l_hidden: 
      - 64
      activation : 
      - relu
      out_activation : relu
  
  x_dim: 512
  z_dim: 64
```

* arch - the parameter that tell which AE architectire should be used. The architectures are listed bellow.
* x_dim - the input vector dimenstion
* z_dim - the dimension of compresseion or last hidden layer

The encoder and decoder are nested structures the parameters of which are as follow:

* arch - architecture of layers. Curently, only `fc_vec` is available.
* linear_layer_type - what type of dimension should be used. It can be either `euclidian` or `hyperbolic`.
* l_hidden - a list of overall hidden layers with corresponding dimenstions. Note that list can be emplty.
* activation - activation functions that follows after each hidden layer.
* out_activation - the activation function after last hiddem layer.

## AE architectures

### Vanila

This is a good old vanila autoencoder. The value of `arch` is `ae`.

**Required dataset class**: EmbeddingDataset

### DCEC

This autoencoder uses additional clustering layer in order to enforce a particular distribution of compressed vectors.

The value `arch` is `ae`. The additional parameter `n_clusters` is required.

**Required dataset class**: EmbeddingDataset

### NRAE

This autoencoder leverage the information about neighbours of each point.

The value `arch` is `nrael`. The additional nested parameter is required 

```yaml
kernel:  
    type: binary
    lambda: 0.5
```

**Required dataset class**: EmbeddingDatasetWithGraph. To initialize the dataset, the next parameters are required:

```yaml
graph_config:
  use_graph : True
  include_center : True
  replace: False
  num_nn: 3
  bs_nn: 2
```

## Default configs

There are several default configs in `configs` directoy.

## Related repositories

1. [Official NRAE repository](https://github.com/Gabe-YHLee/NRAE-public)
2. [DCEC repository](https://github.com/XifengGuo/DCEC)
3. [Hyperbolic (ordinary and variational) autoencoders for recommender systems](https://github.com/evfro/HyperbolicRecommenders)
4. [Hyperbolic Image Embeddings](https://github.com/leymir/hyperbolic-image-embeddings)