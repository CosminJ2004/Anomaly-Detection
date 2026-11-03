import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pyod.utils.data import generate_data_clusters
from pyod.utils.utility import standardizer
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.combination import average, maximization

np.random.seed(42)


def leverage_scores(X):
    # h_i = diagonal of the hat matrix H = X (X^T X)^-1 X^T
    # with X = QR we get H = Q Q^T, so h_i is just the squared norm of row i of Q
    Q, _ = np.linalg.qr(X)
    return np.sum(Q ** 2, axis=1)


def scatter_labels(ax, X, y, title):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", s=10, label="inliers")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:red", s=10, label="outliers")
    ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------- Ex. 1
print("=" * 60)
print("Ex. 1 - leverage scores")
print("=" * 60)

n_reg = 100     # regular points
n_odd = 10      # points of each "noisy" type
top_k = 10      # how many of the highest leverage points we mark
types = ["regular", "high var x", "high var y", "high var x+y"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# the (mu, sigma^2) of the noise added to the 3 noisy groups
mus = [0, 3]
sigma2s = [1, 4, 16]


def make_group(n, n_feat, coefs, mu, sigma2, noisy_x, noisy_y):
    X = np.random.uniform(-3, 3, size=(n, n_feat))
    # regular points: small noise, so they stay close to the model
    y = X @ coefs[:-1] + coefs[-1] + np.random.normal(0, 0.3, n)
    if noisy_x:
        X = X + np.random.normal(mu, np.sqrt(sigma2), size=(n, n_feat))
    if noisy_y:
        y = y + np.random.normal(mu, np.sqrt(sigma2), n)
    return X, y


def make_dataset(n_feat, coefs, mu, sigma2):
    Xs, ys, groups = [], [], []
    flags = [(False, False), (True, False), (False, True), (True, True)]
    for g, (fx, fy) in enumerate(flags):
        n = n_reg if g == 0 else n_odd
        X, y = make_group(n, n_feat, coefs, mu, sigma2, fx, fy)
        Xs.append(X)
        ys.append(y)
        groups.append(np.full(n, g))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(groups)


for n_feat in [1, 2]:
    # random linear model: y = a*x1 (+ b*x2) + c + eps
    coefs = np.random.uniform(-2, 2, size=n_feat + 1)
    print(f"\n{n_feat}D model, coefficients (last one = intercept): {np.round(coefs, 3)}")

    fig = plt.figure(figsize=(4 * len(sigma2s), 3.6 * len(mus)))
    for i, mu in enumerate(mus):
        for j, sigma2 in enumerate(sigma2s):
            X, y, groups = make_dataset(n_feat, coefs, mu, sigma2)

            # design matrix = features + a column of ones for the intercept
            A = np.column_stack([X, np.ones(len(X))])
            h = leverage_scores(A)
            top = np.argsort(h)[-top_k:]

            # sanity check: sum of leverages = number of parameters
            mean_h = [h[groups == g].mean() for g in range(4)]
            print(f"  mu={mu}, sigma2={sigma2:>2}: sum(h)={h.sum():.2f} | mean h per group: "
                  + ", ".join(f"{t}={m:.3f}" for t, m in zip(types, mean_h))
                  + f" | top-{top_k} groups: {np.bincount(groups[top], minlength=4)}")

            idx = i * len(sigma2s) + j + 1
            if n_feat == 1:
                ax = fig.add_subplot(len(mus), len(sigma2s), idx)
                for g in range(4):
                    ax.scatter(X[groups == g, 0], y[groups == g], c=colors[g], s=10, label=types[g])
                ax.scatter(X[top, 0], y[top], s=80, facecolors="none", edgecolors="k",
                           label=f"top {top_k} leverage")
                xs = np.linspace(X.min(), X.max(), 10)
                ax.plot(xs, coefs[0] * xs + coefs[1], "k--", lw=1)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            else:
                ax = fig.add_subplot(len(mus), len(sigma2s), idx, projection="3d")
                for g in range(4):
                    ax.scatter(X[groups == g, 0], X[groups == g, 1], y[groups == g],
                               c=colors[g], s=8, label=types[g])
                ax.scatter(X[top, 0], X[top, 1], y[top], s=80, facecolors="none",
                           edgecolors="k", label=f"top {top_k} leverage")
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                ax.set_zlabel("y")
            ax.set_title(f"mu={mu}, sigma^2={sigma2}", fontsize=9)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.suptitle(f"Ex. 1 - leverage scores, {n_feat}D model")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f"ex1_leverage_{n_feat}d.png", dpi=120)

# ---------------------------------------------------------------- Ex. 2
print()
print("=" * 60)
print("Ex. 2 - KNN on generate_data_clusters")
print("=" * 60)

contamination = 0.1
X_train, X_test, y_train, y_test = generate_data_clusters(
    n_train=400, n_test=200, n_clusters=2, n_features=2,
    contamination=contamination, random_state=42)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

ks = [1, 5, 10, 30, 60]
fig, axes = plt.subplots(len(ks), 4, figsize=(14, 3.2 * len(ks)))
for r, k in enumerate(ks):
    clf = KNN(n_neighbors=k, contamination=contamination)
    clf.fit(X_train)
    y_train_pred = clf.labels_          # same as clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    ba_train = balanced_accuracy_score(y_train, y_train_pred)
    ba_test = balanced_accuracy_score(y_test, y_test_pred)
    print(f"n_neighbors={k:>2}: BA train = {ba_train:.3f}, BA test = {ba_test:.3f}")

    scatter_labels(axes[r, 0], X_train, y_train, f"k={k} train - ground truth")
    scatter_labels(axes[r, 1], X_train, y_train_pred, f"k={k} train - predicted (BA={ba_train:.2f})")
    scatter_labels(axes[r, 2], X_test, y_test, f"k={k} test - ground truth")
    scatter_labels(axes[r, 3], X_test, y_test_pred, f"k={k} test - predicted (BA={ba_test:.2f})")

axes[0, 0].legend(fontsize=8)
fig.suptitle("Ex. 2 - KNN for different n_neighbors")
fig.tight_layout()
fig.savefig("ex2_knn.png", dpi=110)

# ---------------------------------------------------------------- Ex. 3
print()
print("=" * 60)
print("Ex. 3 - KNN vs LOF on clusters with different densities")
print("=" * 60)

X3, c3 = make_blobs(n_samples=[200, 100], n_features=2,
                    centers=[(-10, -10), (10, 10)], cluster_std=[2, 6],
                    random_state=42)
cont3 = 0.07

ks3 = [5, 10, 20, 40]
fig, axes = plt.subplots(len(ks3), 2, figsize=(9, 4.2 * len(ks3)))
for r, k in enumerate(ks3):
    for c, (name, model) in enumerate([("KNN", KNN), ("LOF", LOF)]):
        clf = model(n_neighbors=k, contamination=cont3)
        clf.fit(X3)
        pred = clf.labels_
        # how many of the flagged points come from the dense / sparse cluster
        n_dense = int(pred[c3 == 0].sum())
        n_sparse = int(pred[c3 == 1].sum())
        print(f"  {name} k={k:>2}: outliers in dense cluster = {n_dense:>2}, in sparse cluster = {n_sparse:>2}")
        scatter_labels(axes[r, c], X3, pred,
                       f"{name}, k={k} (dense: {n_dense}, sparse: {n_sparse})")

fig.suptitle("Ex. 3 - KNN vs LOF")
fig.tight_layout()
fig.savefig("ex3_knn_vs_lof.png", dpi=110)

# ---------------------------------------------------------------- Ex. 4
print()
print("=" * 60)
print("Ex. 4 - ensembles of KNN / LOF on cardio")
print("=" * 60)

data = loadmat("cardio.mat")
X4, y4 = data["X"], data["y"].ravel().astype(int)
cont4 = y4.mean()
print(f"cardio: {X4.shape}, contamination = {cont4:.4f}")

X4_train, X4_test, y4_train, y4_test = train_test_split(
    X4, y4, test_size=0.3, stratify=y4, random_state=42)

# mean/std computed on train only, then applied to test as well
X4_train, X4_test = standardizer(X4_train, X4_test)

neighbors = np.linspace(30, 120, 10).astype(int)

for name, model in [("KNN", KNN), ("LOF", LOF)]:
    print(f"\n{name} models:")
    train_scores = np.zeros((len(X4_train), len(neighbors)))
    test_scores = np.zeros((len(X4_test), len(neighbors)))

    for i, k in enumerate(neighbors):
        clf = model(n_neighbors=k, contamination=cont4)
        clf.fit(X4_train)
        ba_train = balanced_accuracy_score(y4_train, clf.labels_)
        ba_test = balanced_accuracy_score(y4_test, clf.predict(X4_test))
        print(f"  n_neighbors={k:>3}: BA train = {ba_train:.3f}, BA test = {ba_test:.3f}")

        train_scores[:, i] = clf.decision_scores_
        test_scores[:, i] = clf.decision_function(X4_test)

    # each model has scores on a different scale, so we normalize them
    # (again with the mean/std from train) before combining
    train_norm, test_norm = standardizer(train_scores, test_scores)

    for strategy, combine in [("average", average), ("maximization", maximization)]:
        train_comb = combine(train_norm)
        test_comb = combine(test_norm)
        # threshold = the score above which the top cont4 fraction of train points are
        thr = np.quantile(train_comb, 1 - cont4)
        ba_train = balanced_accuracy_score(y4_train, (train_comb > thr).astype(int))
        ba_test = balanced_accuracy_score(y4_test, (test_comb > thr).astype(int))
        print(f"  {name} ensemble - {strategy:<12}: threshold = {thr:.3f}, "
              f"BA train = {ba_train:.3f}, BA test = {ba_test:.3f}")

plt.show()
