The goal of this project is to test the influence of two conditional probability function characteristics -- Hölder smoothness and the small value bound -- on neural network classification performance. It is somehow also pretty unique in that said "performance" refers to conditional probability approximation rather than most probable class prediction. (That may be because most images do not display 0.8 dogs and 0.2 cats.)

### `simulating.py`
We sample (multi-dimensional) features from some specified distribution. These features serve as input for some specified functions, one for every output class. The outputs are normalized, resulting in a conditional probability vector per sample. That serves as probability vector of a categorical distribution from which the final label/class of this sample is drawn. 

### `training.py`
The networks are trained with the features and the one-hot encoded label (which may not be the most likely one!) -- we will see the conditional probability functions back later. The network implementation is fairly straightforward Keras on TensorFlow, though we want to be able to easily modify some network parameters like the number and widths of layers and L1 penalty. As is common anyway but essential here, ReLU activation functions are used throughout while the final layer's is the softmax.

### `evaluating.py`
We are primarily interested in the Küllback-Leibler divergence between the true conditional class probabilities and the predicted ones. (More specifically, a truncated version that essentially puts a bound on a very small log(true/predicted).) Again, training did _not_ use the true probabilities -- only the categorical one-hot encoding. Furthermore, it used a somewhat different loss, namely categorical cross-entropy / negative log likelihood. 

This truncated KL divergence is interesting because it is theoretically boundable given assumptions on the specified stuff and cross-entropy training loss. A few of these bounds given slightly different situations have been proven by one of the supervisors of this project. This is an attempt to confirm them somewhat empirically. 

The non-constant, interesting part of these bounds has to do with the "convergence rate", which is something like `n ** (-2 * β / (2 * β + d))` where n is the sample size, d the number of features, and β the Hölder smoothness-index. The proportion of small probabilities (α-small value bound) also features in the exponent of results but it is less Gooogleable and also not too important here.

The goal is to recover this rate in some experiments. The choice of feature distribution and conditional probability functions is pretty theoretical -- something like a uniform on [0, 1] as feature distribution makes manual calculation of α and β easiest. I think I will let network parameters be found with Bayesian optimization per simulation situation and even sample size. 
