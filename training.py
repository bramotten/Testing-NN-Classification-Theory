import tensorflow as tf
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Dropout

from basic_imports import *


def keras_prep(X, Y_prob):
    if X.ndim == 1:
        X = X.reshape(-1, 1)  # make 2-dimensional

    _, k = Y_prob.shape
    # Assign sample to category with a categorical probability.
    Y_cat = np.array([np.random.choice(k, 1, p=p) for p in Y_prob])
    Y_one_hot = np.eye(k)[Y_cat.reshape(-1,)]  # now a one-hot matrix
    return X, Y_one_hot


def keras_classifier(hidden_widths, X_train, Y_train, drop=.1, l1=.001):
    # ReLU activations for hidden layer, softmax for final.
    # Force _some_ sparsity through dropout, penalize weights with L1.
    # Note: not the same as theory yet, that has normalization and probably
    #       requires some special strategies for _real_ sparsity.

    p = [X_train[0].shape, *hidden_widths, Y_train[0].size]
    L = len(p) - 2
    model = tf.keras.models.Sequential(name=f'L-is-{L}-and-p_0-is-{p[0][0]}')

    model.add(Dense(p[1], input_shape=p[0], name=f'p_0->p_1',
                    kernel_regularizer=regularizers.l1(l1),
                    bias_regularizer=regularizers.l1(l1)))
    for i in range(2, len(p) - 1):
        model.add(Dropout(drop))
        model.add(Dense(p[i], activation='relu', name=f'p_{i-1}->p_{i}',
                        kernel_regularizer=regularizers.l1(l1),
                        bias_regularizer=regularizers.l1(l1)))

    # Note: no dropout here.
    model.add(Dense(p[-1], activation='softmax', name=f'p_{L}->p_{L+1}'))
    return model


def train_network(X, Y_prob, test_prop=0.2, hidden_widths=[16, 16, 32, 16, 16],
                  loss_fn='categorical_crossentropy', optimizer='adam',
                  viz=0, val_s=.20):
    X, Y_one_hot = keras_prep(X, Y_prob)

    n_train = int(np.floor(X.shape[0] * (1 - test_prop)))
    X_train = X[:n_train]
    X_test = X[n_train:]
    Y_train = Y_one_hot[:n_train]
    Y_test = Y_one_hot[n_train:]
    # Y_prob_train = Y_prob[:n_train]  # probably never useful.
    Y_prob_test = Y_prob[n_train:]

    model = keras_classifier(hidden_widths, X_train, Y_train)

    if viz > 0:
        print("Max 0/1-accuracy during training:",
              np.mean(Y_prob.argmax(axis=1) == Y_one_hot.argmax(axis=1)))
        print(model.summary())

    model.compile(optimizer, loss_fn, metrics=['accuracy'])
    cb = [tf.keras.callbacks.EarlyStopping('loss', min_delta=.001,
                                           patience=10, verbose=1,
                                           restore_best_weights=True)]
    history = model.fit(X_train, Y_train, epochs=420, validation_split=val_s,
                        callbacks=cb, batch_size=12, use_multiprocessing=True,
                        verbose=viz-1) # TODO: check if last argument works.

    if viz > 0:
        pd.DataFrame(history.history).plot()
        plt.gca().set_ylim(0, max(1, np.quantile(history.history['val_loss'],
                                                 .98)))
        # ^ Dynamic to make it compatible with different loss functions.
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.show()

    return model, X_test, Y_test, Y_prob_test
