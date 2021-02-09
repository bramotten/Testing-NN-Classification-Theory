The goal of this project is to test the influence of two conditional probability function characteristics -- Hölder smoothness and the small value bound -- on neural network classification performance.

### `simulating.py`
We sample (multi-dimensional) features from some specified distribution. These features serve as input for some specified functions, one for every output class. The outputs are normalized for a conditional probability vector per sample. This then serves as probability vector of a categorical distribution from which the final label/class of this sample is drawn. 

### `training.py`
The networks are trained with the features and the one-hot encoded output (which may not be the most likely label!) -- we will use the conditional probability functions again later. This training is fairly straightforward Keras on TensorFlow, though we want to be able to easily modify some network parameters like the number and widths of layers and L1 penalty.

### `evaluating.py`

