from sklearn.calibration import calibration_curve
import tensorflow as tf

from .basic_imports import *


def visualize(model, X_test, Y_test, Y_prob_test):
    Y_test_pred = model.predict(X_test)  # probabilities ofc.
    print("Some examples from test set:")
    print("X:")
    print(X_test.round(3))
    print("True:")
    print(Y_prob_test.round(3))
    print("Predict:")
    print(Y_test_pred.round(3))

    if X_test.shape[1] == 1:
        order = np.argsort(X_test[:, 0])
        for i in range(Y_prob_test.shape[1]):
            plt.plot(X_test[order], Y_test_pred[order, i], label=f'$\hat{{p}}_{i+1}(x)$')
            plt.plot(X_test[order], Y_prob_test[order, i], '--', label=f'$p^0_{i+1}(x)$',
                     color=plt.gca().lines[-1].get_color())
        plt.xlabel('x')
        plt.legend()
        plt.show()

        # MAYDO: remove next calibration stuff, then don't need y_test
        if Y_test[0].size == 2:

            prob_sec = model.predict(X_test)[:, 1]
            n_bins = 20
            name = f'Network {model.name}'
            frac_sec, mean_pred = calibration_curve(Y_test.argmax(axis=1), prob_sec, n_bins=n_bins)

            plt.figure(figsize=(12, 10))
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))

            ax1.plot([0, 1], [0, 1], label="Perfectly calibrated")
            ax1.plot(mean_pred, frac_sec, label=name)
            ax2.hist(prob_sec, range=(0, 1), bins=n_bins, label=name, density=True)

            ax1.set_ylabel("Fraction of second class predictions")
            ax1.set_ylim([-0.05, 1.05])
            ax1.legend(loc="lower right")
            ax1.set_title('Calibration plot (reliability curve)')

            ax2.set_xlabel("Mean predicted value")
            ax2.set_ylabel("Density")
            ax2.legend(ncol=2)

            plt.tight_layout()
            plt.show()
    elif X_test.shape[1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        n, k = Y_prob_test.shape
        subset = np.random.choice(n, 200)
        step = int(n / 200)
        subset = list(np.argsort(X_test[:, 0]))[::step]
        # MAYDO: seperate plots for both classes
        for i in range(k):
            ax.scatter(X_test[subset, 0], X_test[subset, 1], Y_test_pred[subset, i],
                       label=f'$\hat{{p}}_{i+1}(\mathbf{{x}})$')
            ax.scatter(X_test[subset, 0], X_test[subset, 1], Y_prob_test[subset, i],
                       label=f'$p^0_{i+1}(\mathbf{{x}})$', marker='^')

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.legend()
        plt.show()
        plt.show()
    else:
        print("Not visualizing since X.ndim > 2")


def ll_loss(true, pred):
    return tf.keras.losses.categorical_crossentropy(true, pred).numpy().mean()


def KL_loss(true, pred):
    return tf.keras.losses.KLDivergence()(true, pred).numpy().mean()


def KL_trunc_loss(true, pred, B, minimum=1e-5):
    # Ugly to make it transparent
    divergences = []
    for i, true_vec_i in enumerate(true):
        pred_vec_i = pred[i]
        current_sum = 0
        for k, true_i_k in enumerate(true_vec_i):
            pred_i_k = pred_vec_i[k]
            if pred_i_k < minimum:
                current_sum += true_i_k * B
            else:
                log_true_pred_ratio = np.log(true_i_k / pred_i_k)
                if B < log_true_pred_ratio:
                    current_sum += true_i_k * B
                else:
                    current_sum += true_i_k * log_true_pred_ratio
        divergences.append(current_sum)
    return np.mean(divergences)


def MSE(true, pred):
    return tf.keras.losses.MeanSquaredError()(true, pred).numpy()


def test_loss(model, X_test, Y_test, Y_prob_test, B=1.5, Y_test_pred=None):
    if Y_test_pred is None:
        Y_test_pred = model.predict(X_test)
    return {
        "One-hot LL  ": ll_loss(Y_test, Y_test_pred),
        "One-hot MSE ": MSE(Y_test, Y_test_pred),
        "One-hot KL  ": KL_loss(Y_test, Y_test_pred),
        "Probab. LL  ": ll_loss(Y_prob_test, Y_test_pred),
        "Probab. MSE ": MSE(Y_prob_test, Y_test_pred),
        "Probab. KL  ": KL_loss(Y_prob_test, Y_test_pred),
        f"Pr. KL_B={B}": KL_trunc_loss(Y_prob_test, Y_test_pred, B)
    }


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
    return {
        "Biases > e. ": nz_biases, "Total biases": biases,
        "Weights > e.": nz_weights, "Tot. weights": weights,
        "Epsilon     ": epsilon, "s           ": nz_biases + nz_weights
    }


def sup_prob_diff(probs1, probs2):
    return np.amax(np.abs(probs1 - probs2))


def get_all_quantities_of_interest(model, X_test, Y_test, Y_prob_test):
    Y_test_pred = model.predict(X_test)
    return {
        **test_loss(model, X_test, Y_test, Y_prob_test, Y_test_pred=Y_test_pred),
        **get_sparsity(model),
        "Pr. max difference": sup_prob_diff(Y_prob_test, Y_test_pred)
    }
