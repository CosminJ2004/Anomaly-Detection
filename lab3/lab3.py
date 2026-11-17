import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.utils.utility import standardizer
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

np.random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))


def out(name):
    return os.path.join(HERE, name)


# ---------------------------------------------------------------- Ex. 1
print("=" * 60)
print("Ex. 1 - simple LODA built by hand")
print("=" * 60)

X, _ = make_blobs(n_samples=500, n_features=2, centers=[(0, 0)], cluster_std=1.0,
                  random_state=42)
n_proj = 5
# random directions from N(0, I), then normalized to unit length
W = np.random.multivariate_normal([0, 0], np.eye(2), size=n_proj)
W /= np.linalg.norm(W, axis=1, keepdims=True)

X_test = np.random.uniform(-3, 3, size=(500, 2))


def simple_loda(X_train, X_test, W, n_bins):
    scores = np.zeros(len(X_test))
    for w in W:
        z = X_train @ w
        # range wider than the projected data, so test points that fall outside
        # the training range still land in an (empty) bin instead of being dropped
        lo, hi = z.min() - 3, z.max() + 3
        counts, edges = np.histogram(z, bins=n_bins, range=(lo, hi))
        p = counts / counts.sum()
        idx = np.clip(np.digitize(X_test @ w, edges) - 1, 0, n_bins - 1)
        scores += p[idx]
    return scores / len(W)


bins_list = [5, 10, 20, 50, 100]
fig, axes = plt.subplots(1, len(bins_list), figsize=(4 * len(bins_list), 4))
for ax, nb in zip(axes, bins_list):
    s = simple_loda(X, X_test, W, nb)
    sc = ax.scatter(X_test[:, 0], X_test[:, 1], c=s, cmap="viridis", s=12)
    ax.scatter(X[:, 0], X[:, 1], c="r", s=2, alpha=0.3)
    ax.set_title(f"n_bins={nb}")
    print(f"  n_bins={nb:>3}: score min={s.min():.4f}, max={s.max():.4f}")
fig.colorbar(sc, ax=axes, label="mean probability (low = anomalous)")
fig.suptitle("Ex. 1 - simple LODA score map (red = training data)")
fig.savefig(out("ex1_loda_bins.png"), dpi=110, bbox_inches="tight")

# ---------------------------------------------------------------- Ex. 2
print()
print("=" * 60)
print("Ex. 2 - artifacts of IForest vs DIF vs LODA")
print("=" * 60)


def make_models(dim, contamination=0.02, hidden=None, n_bins=10):
    return {
        "IForest": IForest(contamination=contamination, random_state=42),
        "DIF": DIF(contamination=contamination, random_state=42,
                   **({"hidden_neurons": hidden} if hidden else {})),
        "LODA": LODA(contamination=contamination, n_bins=n_bins),
    }


def score_map(dim, centers, models_fn, title, fname):
    Xtr, _ = make_blobs(n_samples=[500, 500], n_features=dim, centers=centers,
                        cluster_std=1, random_state=42)
    Xte = np.random.uniform(-10, 20, size=(1000, dim))
    models = models_fn()
    fig = plt.figure(figsize=(5 * len(models), 4.5))
    for i, (name, m) in enumerate(models.items(), 1):
        m.fit(Xtr)
        s = m.decision_function(Xte)
        if dim == 2:
            ax = fig.add_subplot(1, len(models), i)
            sc = ax.scatter(Xte[:, 0], Xte[:, 1], c=s, cmap="viridis", s=10)
        else:
            ax = fig.add_subplot(1, len(models), i, projection="3d")
            sc = ax.scatter(Xte[:, 0], Xte[:, 1], Xte[:, 2], c=s, cmap="viridis", s=10)
        ax.set_title(name)
        fig.colorbar(sc, ax=ax, shrink=0.7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out(fname), dpi=110)


c2 = [(10, 0), (0, 10)]
c3 = [(0, 10, 0), (10, 0, 10)]

print("  2D, default parameters")
score_map(2, c2, lambda: make_models(2), "Ex. 2 - 2D, default", "ex2_2d.png")
print("  3D, default parameters")
score_map(3, c3, lambda: make_models(3), "Ex. 2 - 3D, default", "ex2_3d.png")

# different DIF hidden layers / LODA bins (2D)
for hidden in [[16], [64, 32], [128, 64, 32]]:
    for nb in [5, 20, 100]:
        print(f"  2D, DIF hidden={hidden}, LODA bins={nb}")
        score_map(2, c2, lambda: make_models(2, hidden=hidden, n_bins=nb),
                  f"2D, DIF hidden={hidden}, LODA n_bins={nb}",
                  f"ex2_2d_h{'-'.join(map(str, hidden))}_b{nb}.png")

# ---------------------------------------------------------------- Ex. 3
print()
print("=" * 60)
print("Ex. 3 - shuttle: IForest vs LODA vs DIF")
print("=" * 60)

path = out("shuttle.mat")
if not os.path.exists(path):
    print("shuttle.mat not found in lab3/ - download it from ODDS and re-run.")
else:
    data = loadmat(path)
    Xs, ys = data["X"], data["y"].ravel().astype(int)
    cont = ys.mean()
    print(f"shuttle: {Xs.shape}, contamination = {cont:.4f}")

    res = {"IForest": [], "LODA": [], "DIF": []}
    for seed in range(10):
        Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.4, random_state=seed)
        Xtr, Xte = standardizer(Xtr, Xte)
        models = {
            "IForest": IForest(contamination=cont, random_state=seed),
            "LODA": LODA(contamination=cont),
            "DIF": DIF(contamination=cont, random_state=seed),
        }
        for name, m in models.items():
            m.fit(Xtr)
            ba = balanced_accuracy_score(yte, m.predict(Xte))
            auc = roc_auc_score(yte, m.decision_function(Xte))
            res[name].append((ba, auc))
        print(f"  split {seed}: " + ", ".join(
            f"{n} BA={r[-1][0]:.3f} AUC={r[-1][1]:.3f}" for n, r in res.items()))

    print("\nMean over 10 splits:")
    for name, r in res.items():
        r = np.array(r)
        print(f"  {name:<8}: BA = {r[:, 0].mean():.3f} +- {r[:, 0].std():.3f}, "
              f"ROC AUC = {r[:, 1].mean():.3f} +- {r[:, 1].std():.3f}")
