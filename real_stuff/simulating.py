from .basic_imports import *


def visualize(X, Y_prob, fX=False):
    print("X:\n", X.round(3))
    print("Normalized funcs(X) = Y probabilities:")
    print(Y_prob.round(3))
    if type(fX) != bool:
        print("Unnormalized funcs(X):")
        print(fX.round(3))
        print("Euclidean distance between those:", np.linalg.norm(fX - Y_prob))

    t_space = np.geomspace(1e-20, 1, 10_000)  # denser where small.
    p_X_smaller = [np.mean(Y_prob <= t) for t in t_space]

    if X.ndim > 2:
        print("X dimensionality too high to visualize.")
        return -1

    plt.figure()
    plt.xlabel("$x$")
    plt.ylim(0, 1.05)

    if X.ndim == 1:
        plt.plot(t_space, p_X_smaller, color='red', label='$\mathbb{P}(\mathbf{p}(X) \leq x)$')
        x, _, p = plt.hist(X, bins=30, density=True, alpha=.3, label='Scaled density of $X$')
        # Histogram scaling:
        for item in p:
            item.set_height(item.get_height() / max(x))

        # Empirical function plotting:
        order = np.argsort(X)
        for i in range(Y_prob.shape[1]):
            plt.plot(X[order], Y_prob[order, i], label=f'$p^0_{i+1}(x)$')

        plt.legend()
        plt.show()
    elif X.ndim == 2:
        plt.plot(t_space, p_X_smaller, color='red',
                 label='$\mathbb{P}(\mathbf{p}(\mathbf{X}) \leq x)$')
        x, _, p = plt.hist(X, bins=30, density=True, alpha=.3,
                           label=[f'Scaled density of $X_{j+1}$' for j in range(X.shape[1])])
        for j in range(X.ndim):
            for item in p[j]:
                item.set_height(item.get_height() / max(x[j]))

        plt.legend()
        plt.show()

        # Subset in case 2D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        subset = np.random.choice(X.shape[0], 200)
        for i in range(Y_prob.shape[1]):
            ax.scatter(X[subset, 0], X[subset, 1], Y_prob[subset, i],
                       label=f'$p^0_{i+1}(\mathbf{{x}})$')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.legend()
        plt.show()
        plt.show()


def unif_rejection_sampling(p, n=10_000, seed=1):
    # Rejection sampling from p; proposal is unif[0, 1].
    # MAYDO: faster sampling method.
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


def normalize(fX):
    # fX's shape is n x len(funcs).
    return np.array([fX[i] / x_sum for i, x_sum in enumerate(fX.sum(axis=1))])


def probs(X, funcs, viz=False):
    fX = np.array([f(X) for f in funcs]).T
    Y_prob = normalize(fX)
    for probability_vector in Y_prob:
        assert min(probability_vector) > 0  # TODO: ask if we want this
    if viz:
        visualize(X, Y_prob, fX)
    return Y_prob


def sample_and_funcs(situation, n=10_000, K=False, viz=False, seed=42):
    if K and "6." not in situation:
        print("Can't modify K in this situation (yet).")
    np.random.seed(seed)
    # NOTE: deviating from .docx situations now
    if situation == "1":
        desc = f"Situation 1: sampling {n} X_i ~ 1D uniform. f1(X) = (1 + X) / 3. K = 2."
        X = np.random.uniform(size=n)
        def f1(X): return (1 + X) / 3
        def f2(X): return (2 - X) / 3
        funcs = [f1, f2]
    elif situation == "2":
        desc = f"Situation 2: sampling {n} X_i ~ 2D uniforms. f1(X) = sum(X) / 2. K = 2."
        X1 = np.random.uniform(size=n)
        X2 = np.random.uniform(size=n)
        X = np.column_stack([X1, X2])
        def f1(X): return (X[:, 0] + X[:, 1]) / 2
        def f2(X): return 1 - f1(X)
        funcs = [f1, f2]
    elif situation == "3":
        desc = f"Situation 3: sampling {n} X_i ~ a mixture of normals. K = 2."
        mu = [.4, .8]
        sigma = [.5, .1]
        pY = [.5, .5]

        def p(x):
            return sum([ss.norm(mu[i], sigma[i]).pdf(x) * pY[i] for i in range(len(mu))])

        X = unif_rejection_sampling(p, n, seed)
        funcs = [ss.norm(mu[i], sigma[i]).pdf for i in range(len(mu))]
    elif situation == "4":
        desc = f"Situation 4: sampling {n} X_i ~ 1D uniform. f1(X)=f2(X). K = 2."
        X = np.random.uniform(size=n)
        def f1(X): return (1 + X) / 3
        def f2(X): return f1(X)
        funcs = [f1, f2]
    elif situation == "5":
        m = 15
        desc = f"Sit 5: sampling {n} X_i ~ 1D uniform. f1(X) = X ** {m}. K = 2. Alpha = 1 / m."
        X = np.random.uniform(size=n)
        def f1(X): return X ** m
        def f2(X): return 1 - f1(X)
        funcs = [f1, f2]
    elif "6." in situation:
        # TODO: handle above?
        K_str = situation.split('.')[1]
        if K_str != "":
            K = int(K_str)
        if K == False:
            print("No K specified, setting it to 3.")
            K = 3
        desc = f"Situation 6: sampling {n} X_i ~ 1D uniform. f_k(X) = a normal. K = {K}."
        X = np.random.uniform(size=n)
        funcs = []
        for k in range(K):
            funcs.append(lambda x, k=k: ss.norm(k / K, 0.5 / K).pdf(x))
    elif situation == "7":
        desc = f"Situation 7: sampling {n} X_i ~ 1D uniform. f1(X) = TODO. K = 2."
        X = np.random.uniform(size=n)
        beta = 4
        def f1(X): return 1 / (1 + np.power(X / (1 - X), - 1 * beta))
        def f2(X): return 1 - f1(X)
        funcs = [f1, f2]
    elif situation == "8":
        desc = f"Situation 8: sampling {n} X_i ~ a mixture of uniforms. K = 2."
        a = [0.05, .55]
        b = [.45, .95]
        pY = [.5, .5]

        def p(x):
            return sum([ss.uniform(a[i], b[i]).pdf(x) * pY[i] for i in range(len(pY))])

        X = unif_rejection_sampling(p, n, seed)
        beta = 3
        def f1(X): return 1 / (1 + np.power(X / (1 - X), - 1 * beta))
        def f2(X): return 1 - f1(X)
        funcs = [f1, f2]
    elif situation == "9":
        desc = "Taylor sin(uniform[0, pi / 2])."
        X = np.random.uniform(high=np.math.pi/2, size=n)
        m = 10  # can't be high, factorial gets too large
        def f1(X): return 1e-10 + sum(
            [(-1) ** (i+1) * X ** (2*i-1) / np.math.factorial(2*i-1) for i in range(1, m)])
        def f2(X): return 1 - f1(X)
        funcs = [f1, f2]
    else:
        raise ValueError("Situation not implemented.")

    # MAYDO: return some other trivia relevant for evaluations (alpha, beta?)
    if viz:
        print(desc)
    return X, funcs


def create_dataset(situation, n=10_000, K=False, viz=False, seed=42):
    X, funcs = sample_and_funcs(situation, n, K, viz, seed)
    return X, funcs, probs(X, funcs, viz)
