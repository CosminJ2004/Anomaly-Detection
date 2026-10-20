import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

np.random.seed(42)


def balanced_accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return (tpr + tnr) / 2, (tn, fp, fn, tp)


def print_metrics(name, y_true, y_pred):
    ba, (tn, fp, fn, tp) = balanced_accuracy(y_true, y_pred)
    print(f"  [{name}] TN={tn} FP={fp} FN={fn} TP={tp} | BA={ba:.3f}")
    return ba


# ---------------------------------------------------------------- Ex. 1
print("=" * 60)
print("Ex. 1 - generate data and plot it")
print("=" * 60)

contamination = 0.1
X_train, X_test, y_train, y_test = generate_data(
    n_train=400, n_test=100, n_features=2,
    contamination=contamination, random_state=42)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"outliers in train: {int(y_train.sum())} / {len(y_train)}")
print(f"outliers in test:  {int(y_test.sum())} / {len(y_test)}")

plt.figure(figsize=(6, 5))
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
            c="tab:blue", s=15, label="inliers")
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            c="tab:red", s=15, label="outliers")
plt.title("Ex. 1 - training data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.tight_layout()
plt.savefig("ex1_train_data.png", dpi=120)

# ---------------------------------------------------------------- Ex. 2
print()
print("=" * 60)
print("Ex. 2 - KNN from pyod")
print("=" * 60)

clf = KNN(contamination=contamination)
clf.fit(X_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print(f"contamination = {contamination}, threshold = {clf.threshold_:.4f}")
print_metrics("train", y_train, y_train_pred)
print_metrics("test ", y_test, y_test_pred)

# ROC on the test set using the raw outlier scores
test_scores = clf.decision_function(X_test)
fpr, tpr, thr = roc_curve(y_test, test_scores)
auc = roc_auc_score(y_test, test_scores)
print(f"  test AUC = {auc:.3f}")

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"KNN (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="random")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Ex. 2 - ROC curve (test)")
plt.legend()
plt.tight_layout()
plt.savefig("ex2_roc.png", dpi=120)

# what happens if the model uses a different contamination rate?
print("\nChanging the contamination used by the model:")
for c in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]:
    m = KNN(contamination=c)
    m.fit(X_train)
    print(f"contamination = {c}")
    print_metrics("train", y_train, m.predict(X_train))
    print_metrics("test ", y_test, m.predict(X_test))

# ---------------------------------------------------------------- Ex. 3
print()
print("=" * 60)
print("Ex. 3 - Z-score on 1D data")
print("=" * 60)

X1, y1 = generate_data(
    n_train=1000, n_test=0, n_features=1,
    contamination=0.1, train_only=True, random_state=42)
X1 = X1.ravel()
print(f"data: {X1.shape}, outliers: {int(y1.sum())}")

z = np.abs((X1 - X1.mean()) / X1.std())
# the top 10% of the z-scores are marked as anomalies
thr = np.quantile(z, 1 - 0.1)
y1_pred = (z > thr).astype(int)
print(f"z-score threshold = {thr:.4f}")
print_metrics("1D z-score", y1, y1_pred)

plt.figure(figsize=(8, 3))
plt.scatter(np.arange(len(X1)), X1, c=np.where(y1_pred == 1, "tab:red", "tab:blue"), s=8)
plt.title("Ex. 3 - 1D data, predicted anomalies in red")
plt.xlabel("sample index")
plt.ylabel("value")
plt.tight_layout()
plt.savefig("ex3_zscore_1d.png", dpi=120)

# ---------------------------------------------------------------- Ex. 4
print()
print("=" * 60)
print("Ex. 4 - Z-score on multidimensional data (Cholesky)")
print("=" * 60)

n = 1000
cont = 0.1
n_out = int(n * cont)
n_in = n - n_out

mu = np.array([2.0, -1.0, 3.0])
Sigma = np.array([[2.0, 0.8, 0.3],
                  [0.8, 1.5, 0.5],
                  [0.3, 0.5, 1.0]])
d = len(mu)

L = np.linalg.cholesky(Sigma)
print("check L @ L.T == Sigma:", np.allclose(L @ L.T, Sigma))

# x ~ N(0, I), then y = L x + mu  (samples are rows, so x @ L.T)
x_in = np.random.randn(n_in, d)
y_in = x_in @ L.T + mu

# anomalies: same shape but scaled up covariance and a shifted mean
x_out = np.random.randn(n_out, d)
y_out = 3 * (x_out @ L.T) + mu + np.array([3.0, 3.0, -3.0])

X4 = np.vstack([y_in, y_out])
y4 = np.concatenate([np.zeros(n_in), np.ones(n_out)]).astype(int)

# shuffle
idx = np.random.permutation(n)
X4, y4 = X4[idx], y4[idx]

# estimate mean and covariance from the data (we don't "know" the true ones)
mu_hat = X4.mean(axis=0)
Sigma_hat = np.cov(X4, rowvar=False)
print("estimated mean:", np.round(mu_hat, 3))

# z-score for multivariate data = Mahalanobis distance. Using the Cholesky
# factor of the estimated covariance: z = L^-1 (y - mu), score = ||z||
L_hat = np.linalg.cholesky(Sigma_hat)
Z = np.linalg.solve(L_hat, (X4 - mu_hat).T).T
scores = np.linalg.norm(Z, axis=1)

thr4 = np.quantile(scores, 1 - cont)
y4_pred = (scores > thr4).astype(int)
print(f"threshold = {thr4:.4f}")
print_metrics("3D z-score", y4, y4_pred)
print(f"  AUC = {roc_auc_score(y4, scores):.3f}")

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection="3d")
ax.scatter(*X4[y4_pred == 0].T, c="tab:blue", s=8, label="predicted normal")
ax.scatter(*X4[y4_pred == 1].T, c="tab:red", s=8, label="predicted anomaly")
ax.set_title("Ex. 4 - 3D data")
ax.legend()
plt.tight_layout()
plt.savefig("ex4_zscore_3d.png", dpi=120)

plt.show()
