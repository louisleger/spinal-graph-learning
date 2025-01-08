import os
import sys
import argparse
import numpy as np
from lib.clustering.glmm import GLMM
from utils import cluster_score, plot_probability_estimates
from visualize_data import visualize_data
from datetime import datetime

def generate_clusters(n = 100, d = 56, radius=1, covariance=None, sigma=1):
    centroid = np.random.normal(size=d); centroid /= np.linalg.norm(centroid)/radius
    if covariance is None: covariance = sigma**2 * np.exp(-(5e-2)*np.abs(np.arange(d)[:, None] - np.arange(d)))

    X = np.concatenate([ 
        np.random.multivariate_normal(np.zeros(d), covariance, n//2),
        np.random.multivariate_normal(centroid, covariance, n//2),
    ])
    labels = np.concatenate([np.zeros(n//2), np.ones(n//2)])
    
    indexes = np.array([np.arange(idx, idx +  n//10) for idx in range(0, n, n//10)])
    np.random.shuffle(indexes)
    indexes = indexes.flatten()

    print("Data shape:", X.shape)
    return X[indexes], labels[indexes], centroid, covariance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=500)
    parser.add_argument('-d', type=int, default=56)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('-s', '--seed', type=int, default=3)
    args = parser.parse_args()

    np.random.seed(args.seed)
    X, labels, centroid, covariance = generate_clusters(n=args.n, d=args.d, radius=args.radius, sigma=args.sigma)
    mixture = GLMM(n_components=2, avg_degree=30, delta=2)
    prediction = mixture.fit_predict(X)
    gamma = mixture.predict_proba(X)
    f1, silhouette = cluster_score(gamma, labels, X, prediction)
    print("Activity Score:", f1)
    print("Silhouette Score:", silhouette)

    execution_command = " ".join(sys.argv)
    current_timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    folder_name = f"data/runs/glmm-synthetic-{current_timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, "command.txt"), "w") as file:
        file.write(f"{execution_command}\n")
        file.write(f"Activity Score: {f1}\n")
        file.write(f"Silhouette Score: {silhouette}\n")

    mean_class_0 = np.zeros(args.d)
    visualize_data(mean_class_0, covariance, path=os.path.join(folder_name, "synthetic_class_0_"), full=False)
    visualize_data(centroid, covariance, path=os.path.join(folder_name, "synthetic_class_1_"), full=False)

    for cluster in range(2):
        visualize_data(mixture.means_[cluster], np.abs(mixture.laplacians_[cluster]), path=os.path.join(folder_name, f"found_cluster_{cluster}_"), full=False)

    plot_probability_estimates(gamma, labels, os.path.join(folder_name, "gamma_plot.png"))

if __name__ == '__main__':
    main()
