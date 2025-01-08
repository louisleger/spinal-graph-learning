import os
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lib.clustering.glmm import GLMM
from utils import cluster_score

warnings.filterwarnings("ignore")

plot_dir = "plots/"
os.makedirs(plot_dir, exist_ok=True)

def generate_clusters(n_clusters=2, d=2, radius=1, angle=0, n=100, plant_cov=None, plant_point=None):
    def rotate_sigma(sigma, angle_rad):
        vals_a, Ua = np.linalg.eigh(sigma)
        n = Ua.shape[1]
        M = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                G = np.eye(n)
                c, s = np.cos(angle_rad), np.sin(angle_rad)
                G[i, i], G[i, j], G[j, i], G[j, j] = c, -s, s, c
                M = M @ G
        Ub = Ua @ M
        return Ub @ np.diag(vals_a) @ Ub.T

    def generate_point(radius, d):
        point = np.random.normal(size=d)
        point /= np.linalg.norm(point) / radius
        return point

    def generate_random_cov(d):
        A = np.random.randn(d, d)
        cov = A @ A.T + 1e-5 * np.eye(d)
        return cov / np.max(np.diag(cov))

    means = [np.zeros(d)] + [generate_point(radius, d) for _ in range(n_clusters - 1)]
    if plant_cov is None:
        sigma = generate_random_cov(d)
        covariances = [sigma] + [rotate_sigma(sigma, np.radians(angle)) for _ in range(n_clusters - 1)]
    else:
        covariances = [plant_cov] + [rotate_sigma(plant_cov, np.radians(angle)) for _ in range(n_clusters - 1)]

    if plant_point is not None:
        means = plant_point

    X = np.concatenate([
        np.random.multivariate_normal(mu, sig, n // n_clusters)
        for mu, sig in zip(means, covariances)
    ])
    labels = np.concatenate([i * np.ones(n // n_clusters) for i in range(n_clusters)])
    idx = np.arange(n)
    np.random.shuffle(idx)
    return X[idx], labels[idx]

radii = np.round(5*np.linspace(0, 3, 15))/5
angles = np.round(np.linspace(0, 60, 15))

f1_mean = np.zeros((len(radii), len(angles)))
f1_std = np.zeros((len(radii), len(angles)))

for i, r in enumerate(radii):
    for j, a in enumerate(angles):
        scores = []
        print("r, a", r, a)
        for _ in range(5):
            X, labels = generate_clusters(n_clusters=2, d=56, radius=r, angle=a, n=2000)
            mixture = GLMM(n_components=2, avg_degree=30, delta=2)
            pred = mixture.fit_predict(X)
            gamma = mixture.predict_proba(X)
            f1, _ = cluster_score(gamma, labels, X, pred)
            scores.append(f1.max())
        f1_mean[i, j] = np.mean(scores)
        f1_std[i, j] = np.std(scores)

plt.figure(figsize=(8, 6))
sns.heatmap(f1_mean, xticklabels=np.round(angles, 2), yticklabels=np.round(radii, 2), annot=True, cmap="viridis")
plt.xlabel("Angle")
plt.ylabel("Radius")
plt.title("Mean F1 Scores")
mean_f1_path = os.path.join(plot_dir, "mean_f1_scores.png")
plt.savefig(mean_f1_path, bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(f1_std, xticklabels=np.round(angles, 2), yticklabels=np.round(radii, 2), annot=True, cmap="viridis")
plt.xlabel("Angle")
plt.ylabel("Radius")
plt.title("Std F1 Scores")
std_f1_path = os.path.join(plot_dir, "std_f1_scores.png")
plt.savefig(std_f1_path, bbox_inches='tight', dpi=300)
plt.close()
