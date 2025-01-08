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
from sklearn.preprocessing import MinMaxScaler


# Run a GLMM algorithm on a specified design matrix and hyperparameters

# EXPL: python run_glmm.py -p database/datasets/C6_GM_spinal_data.npy -k 2 -ag 5


ATLAS = pd.read_csv("data/resources/pam50_atlas.csv")["regions_of_interest"].values
ACTIVITY = np.load("data/resources/activity.npy")
def main():

    parser = argparse.ArgumentParser(description="Run Graph Laplacian Mixture Model")

    parser.add_argument('-p', '--data_path', type=str, help='X data path', required=True)
    parser.add_argument('-k', '--n_components', type=int, help='number of clusters', default=2)
    parser.add_argument('-ag', '--average_degree', type=int, help='Average Degree', default=30)
    parser.add_argument('-d', '--delta', type=float, help='Delta, ratio of beta1 and beta2', default=2)
    parser.add_argument('-f', '--full', type=bool, help='Full Mean and Covariance Plots', default=False)
    parser.add_argument('-s', '--seed', type=int, help='set random seed', default=3)
    parser.add_argument('-n', '--n_subjects', type=int, help='number of subjects concat', default=30)
    parser.add_argument('-pl', '--planted', action="store_true", help='initialize laplacians with functional connectivity', default=False)
    parser.add_argument('-sm', '--smooth', action="store_true", help='smooth gamma probabilities', default=False)
    parser.add_argument('-std', '--standard_dev', action="store_true", help='plot standard dev in gamma plot', default=False)
    args = parser.parse_args()

    X = z_score(np.load(args.data_path)) # (N samples, M dimensions)

    print("Loaded X", X.shape, "mean, std:", X.mean(), X.std())
    duration = X.shape[0]//args.n_subjects

    laplacians_init_ = None
    if args.planted:
        Y = X.reshape(args.n_subjects, duration, 56).mean(axis=0)
        
        laplacians_ = np.stack([np.corrcoef(Y[ACTIVITY==label].T) for label in [0, 1]]) # shape [K, D, D]
        means_ = np.array([Y[ACTIVITY==label].mean(axis=0) for label in [0, 1]]) # shape [K, D]
        resp_ = np.array([(np.tile(ACTIVITY, args.n_subjects) == label).astype(int) for label in [0, 1]]) # shape [K, N]

        laplacians_init_ = (resp_.mean(axis=1), means_,laplacians_, resp_)
        print('init laplacians shape:', laplacians_.shape)

    model = GLMM(n_components=args.n_components, avg_degree=args.average_degree, delta=args.delta, random_state=args.seed, init_params="kmeans", laplacian_init=laplacians_init_)
    
    print("GLMM...")
    prediction = model.fit_predict(X)
    gamma = model.predict_proba(X) 
    print("gamma unique:", np.unique(gamma).shape)

    print("Class weights:", model.weights_)
    f1, silhouette = cluster_score(gamma.reshape(args.n_subjects, duration, args.n_components).mean(axis=0), ACTIVITY[:duration], X, prediction)
    print("Activity Score:", f1)
    print("Silhouette Score:", silhouette)
    
    execution_command = " ".join(sys.argv)
    current_timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    folder_name = f"data/runs/glmm-{current_timestamp}"

    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, "command.txt"), "w") as file:
        file.write(f"{execution_command}\n")
        file.write(f"Activity Score: {f1}\n")
        file.write(f"Silhouette Score: {silhouette}\n")
    

    std = None
    if args.standard_dev:
        std = gamma.reshape(args.n_subjects, duration, args.n_components).std(axis=0)

    gamma = gamma.reshape(args.n_subjects, duration, args.n_components).mean(axis=0)
    plot_probability_estimates(gamma, ACTIVITY[:duration], os.path.join(folder_name, "gamma_plot.png"), gamma_std=std)
    if args.smooth:
        print("Gamma shape", gamma.shape)
        window, passes = 3, 3
        smoothed = gamma
        for _ in range(passes):
            smoothed = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window)/window, mode='valid'), 0, smoothed)
            if args.standard_dev: std = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window)/window, mode='valid'), 0, std)

        mm = MinMaxScaler((0, 1))
        scaled = mm.fit_transform(smoothed)
        #if args.standard_dev: std =  mm.transform(std)
        plot_probability_estimates(scaled, ACTIVITY[(window-1)*passes//2:duration-(window-1)*passes//2], os.path.join(folder_name, "smooth_gamma_plot.png"), gamma_std=std)
    

    for cluster in range(args.n_components): visualize_data(model.means_[cluster], np.abs(model.laplacians_[cluster]), path=f"{folder_name}/k{cluster}_", full=args.full)
    
    print("Done")

if __name__ == '__main__':
    main()
