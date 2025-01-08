import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.clustering.glmm import GLMM
from utils import plot_probability_estimates, cluster_score, z_score
from visualize_data import visualize_data
from datetime import datetime


# Run a GLMM Grid Search algorithm on a specified design matrix and hyperparameters

# EXPL: python run_glmm.py -p database/datasets/C6_GM_spinal_data.npy -k 2 -ag 5

ATLAS = pd.read_csv("data/resources/pam50_atlas.csv")["regions_of_interest"].values
ACTIVITY = np.load("data/resources/activity.npy")


def main():

    parser = argparse.ArgumentParser(description="Run Graph Laplacian Mixture Model")

    parser.add_argument("-p", "--data_path", type=str, help="X data path", required=True)
    parser.add_argument( "-k", "--n_components", type=tuple, help="number of clusters", default=(2, 8))
    parser.add_argument( "-ag", "--average_degree", type=tuple, help="Average Degree", default=(4, 54))
    parser.add_argument("-d", "--delta", type=int, help="Delta, ratio of beta1 and beta2", default=8)
    parser.add_argument("-s", "--seed", type=int, help="set random seed", default=3)
    parser.add_argument("-n", "--n_subjects", type=int, help="number of subjects concat", default=30)
    args = parser.parse_args()

    X = z_score(np.load(args.data_path))  # (N samples, M dimensions)

    print("Loaded X", X.shape, "mean, std:", X.mean(), X.std())
    duration = X.shape[0] // args.n_subjects
    results = {'K':[], 'Degree':[], 'Delta': [], 'F1': [], 'SS': [], 'Weights':[]}
    for k in range(*args.n_components):
        for degree in range(*args.average_degree):
            for delta in 2 ** (np.arange(args.delta) - args.delta/2):
                print("Run...", k, degree, delta)
                model = GLMM(n_components=k, avg_degree=degree, delta=delta,
                                random_state=args.seed,
                                init_params="kmeans",)
                
                prediction = model.fit_predict(X)
                gamma = model.predict_proba(X)
                gamma = gamma.reshape(args.n_subjects, duration, k).mean(axis=0)
                f1, silhouette = cluster_score(gamma, ACTIVITY[:duration], X, prediction)
                results['K'].append(k); results["Degree"].append(degree); results['Delta'].append(delta)
                results['F1'].append(f1); results['SS'].append(silhouette); results['Weights'].append(model.weights_)

    execution_command = " ".join(sys.argv)
    current_timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    folder_name = f"data/runs/glmm-gridsearch{current_timestamp}"

    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, "command.txt"), "w") as file:
        file.write(f"{execution_command}\n")
    
    pd.DataFrame(results).to_csv(folder_name + "/results.csv")
    print("Done")

if __name__ == "__main__":
    main()
