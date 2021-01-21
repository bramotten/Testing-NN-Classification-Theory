from sklearn.calibration import calibration_curve
import tensorflow as tf

from .basic_imports import *


def ll_loss(true, pred):
    return tf.keras.losses.categorical_crossentropy(true, pred).numpy().mean()


def KL_loss(true, pred):
    return tf.keras.losses.KLDivergence()(true, pred).numpy().mean()


def test_loss(model, X_test, Y_test, Y_prob_test, prints=True):
    Y_test_pred = model.predict(X_test)
    losses = {
        "One-hot log-like": ll_loss(Y_test, Y_test_pred),
        "One-hot KL": KL_loss(Y_test, Y_test_pred),
        "Probability vec log-like": ll_loss(Y_prob_test, Y_test_pred),
        "Probability vec KL": KL_loss(Y_prob_test, Y_test_pred),
    }
    if prints:
        print(losses)
    return losses


def visualize(model, X_test, Y_test, Y_prob_test):
    Y_test_pred = model.predict(X_test)
    print("Some examples from test set:")
    print("X:")
    print(X_test.round(3))
    print("True:")
    print(Y_prob_test.round(3))
    print("Predict:")
    print(Y_test_pred.round(3))

    if len(X_test[0]) == 1:
        order = np.argsort(X_test[:, 0])
        for i in range(Y_prob_test.shape[1]):
            plt.plot(X_test[order], Y_test_pred[order, i],
                     label=f'$\hat{{p}}_{{i+1}}(x)$')
            plt.plot(X_test[order], Y_prob_test[order, i], '--',
                     label=f'$p^0_{{i+1}}(x)$')
        plt.xlabel('x')
        plt.legend()
        plt.show()
        if Y_test[0].size == 2:
            # Probaly TODO: remove this stuff, then don't need y_test
            prob_sec = model.predict(X_test)[:, 1]
            n_bins = 20
            name = f'Network {model.name}'
            frac_sec, mean_pred = calibration_curve(Y_test.argmax(axis=1),
                                                    prob_sec, n_bins=n_bins)

            plt.figure(figsize=(12, 10))
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))

            ax1.plot([0, 1], [0, 1], label="Perfectly calibrated")
            ax1.plot(mean_pred, frac_sec, label=name)
            ax2.hist(prob_sec, range=(0, 1), bins=n_bins,
                     label=name, density=True)

            ax1.set_ylabel("Fraction of second class predictions")
            ax1.set_ylim([-0.05, 1.05])
            ax1.legend(loc="lower right")
            ax1.set_title('Calibration plot (reliability curve)')

            ax2.set_xlabel("Mean predicted value")
            ax2.set_ylabel("Density")
            ax2.legend(ncol=2)

            plt.tight_layout()
            plt.show()
    else:
        print("Not visualizing X vs p_k(X) since X > 2-dimensional")
        # TODO


def get_sparsity(model, epsilon=0.001):
    W = model.get_weights()
    nz_biases = biases = nz_weights = weights = 0
    for i, W_i in enumerate(W):
        if i % 2 == 0:
            nz_weights += np.count_nonzero(W_i > epsilon)
            weights += W_i.size
        else:
            nz_biases += np.count_nonzero(W_i > epsilon)
            biases += W_i.size
    return f"Biases > {epsilon}:  {nz_biases}  out of {biases}. \n" + \
           f"Weights > {epsilon}: {nz_weights} out of {weights}."


def get_quantities_of_interest(model):
    # TODO
    pass
