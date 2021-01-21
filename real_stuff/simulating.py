from .basic_imports import *


def normalize(fX):
    # fX's shape is n x len(funcs).
    return np.array([fX[i] / x_sum for i, x_sum in enumerate(fX.sum(axis=1))])


def visualize(X, Y_prob, fX=False):
    print("X:\n", X.round(3))
    if type(fX) != bool:
        print("Unnormalized funcs(X):")
        print(fX.round(3))
    print("Normalized funcs(X) = Y probabilities:")
    print(Y_prob.round(3))
    print("Euclidean distance between those:", np.linalg.norm(fX - Y_prob))

    t_space = np.geomspace(1e-20, 1, 10_000)  # denser where small.
    p_X_smaller = [np.mean(Y_prob <= t) for t in t_space]
    plt.plot(t_space, p_X_smaller, color='red', label='$\mathbb{P}(\mathbf{p}(X) \leq x)$')

    if X.ndim == 1:
        x, _, p = plt.hist(X, bins=30, density=True, alpha=.3, label='Scaled density of $X$')
        # The scaling
        for item in p:
            item.set_height(item.get_height() / max(x))
        order = np.argsort(X)
        for i in range(Y_prob.shape[1]):
            plt.plot(X[order], Y_prob[order, i], label=f'$p^0_{i+1}(x)$')
    else:
        print("2D X visualization is TODO (but I may return some crap already)")
        plt.hist(X, bins=30, density=True, alpha=.3,
                 label=[f'Density of X{j+1}' for j in range(X.shape[1])])

    plt.legend()
    plt.xlabel("x")
    plt.ylim(0, 1.05)
    plt.show()


def unif_rejection_sampling(p, n=10_000, seed=1):
    # Rejection sampling from p; proposal is unif[0, 1].
    # TODO?: better sampling method.
    np.random.seed(seed)
    q_pdf = ss.uniform().pdf
    q_sample = np.random.uniform

    grid = np.linspace(0, 1, 1000)
    m = max(p(grid)) * 1.3  # should divide by q_pdf(grid) but that's always 1.
    # The 1.3 factor is just to be sure m * q > p. Proportion accepted ~ 1/m.

    X = []
    while len(X) < n:
        z = q_sample()
        if np.random.uniform(0, m * q_pdf(z)) <= p(z):
            X.append(z)
    return np.array(X)


def create_dataset(situation, viz=False, seed=42):
    np.random.seed(seed)
    # NOTE: deviating from .docx situations now
    if situation == "1":
        print("Situation 1: sampling 10_000 X_i ~ 1D uniform; f1(X) = (1 + X) / 3.")
        X = np.random.uniform(size=10_000)
        def f1(X): return (1 + X) / 3
        def f2(X): return (2 - X) / 3
        funcs = [f1, f2]
    elif situation == "2":
        print("Situation 2: sampling 10_000 X_i ~ 2D uniforms.")
        X1 = np.random.uniform(size=10_000)
        X2 = np.random.uniform(size=10_000)
        X = np.column_stack([X1, X2])
        def f1(X): return (X[:, 0] + X[:, 1]) / 2
        def f2(X): return 1 - f1(X)
        funcs = [f1, f2]
    elif situation == "3":
        print("Situation 3: sampling 5000 X_i ~ a mixture of normals.")
        mu = [.4, .8]
        sigma = [.5, .1]
        pY = [.5, .5]

        def p(x):
            return sum([ss.norm(mu[i], sigma[i]).pdf(x) * pY[i] for i in range(len(mu))])
        X = unif_rejection_sampling(p, 5000, seed)
        funcs = [ss.norm(mu[i], sigma[i]).pdf for i in range(len(mu))]
    elif situation == "4":
        print("Situation 4: sampling 10_000 X_i ~ 1D uniform; f1(X)=f2(X).")
        X = np.random.uniform(size=10_000)
        def f1(X): return (1 + X) / 3
        def f2(X): return f1(X)
        funcs = [f1, f2]
    elif situation == "5":
        print("Situation 5: sampling 10_000 X_i ~ 1D uniform; f1(X) = X ** 20.")
        X = np.random.uniform(size=10_000)
        def f1(X): return X ** 20
        def f2(X): return 1 - f1(X)
        funcs = [f1, f2]
    else:
        raise ValueError("Situation not implemented.")

    fX = np.array([f(X) for f in funcs]).T
    Y_prob = normalize(fX)
    for probability_vector in Y_prob:
        assert min(probability_vector) > 0  # TODO: ask Thijs if we want this
    if viz:
        visualize(X, Y_prob, fX)

    # MAYDO: return some other trivia for special visualizations
    return X, funcs, Y_prob
