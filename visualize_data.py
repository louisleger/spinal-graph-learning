import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import z_score

# This script allows you to visualize the data from the design matrix X

ATLAS = pd.read_csv("data/resources/pam50_atlas.csv")["regions_of_interest"].values

def get_estimators(X): # X shape (T, N)
    return np.mean(X, axis=0), np.cov(X.T), np.corrcoef(X.T)

def plot_means(mu, labels, name="", path="plots/"):
    fig, ax = plt.subplots(1, 1, figsize = (8, 3))
    
    vrange = max(np.abs(mu))
    norm = plt.Normalize(vmin=-vrange, vmax=vrange, clip=False)
    palette = sns.diverging_palette(230, 20, as_cmap=True)
    colors = palette(norm(mu))

    ax.bar(np.arange(len(mu)), mu, color=colors)
    ax.set_xticks(ticks = np.arange(len(mu)), labels=labels, ha="right", rotation=60, fontsize=6, rotation_mode = "anchor")
    ax.set_xlim(-1, len(mu))
    ax.set_title(f'$\mu${name}')

    plt.tight_layout()
    plt.savefig(f"{path}means{name}.png", bbox_inches = "tight", dpi=200)
    plt.close()

def plot_covariance(cov, labels, name="", path="plots/"):
    fig, ax = plt.subplots(1, 1, figsize = (6, 6))

    sns.heatmap(cov, cmap=sns.diverging_palette(230, 20, as_cmap=True), square=True, center=0, cbar_kws={"shrink": .5}, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(f'$\Sigma${name}', fontsize=10)

    plt.xticks(fontsize = 6); plt.yticks(fontsize = 6)
    plt.tight_layout()
    plt.savefig(f"{path}cov{name}.png", bbox_inches = "tight", dpi=200)
    plt.close()

def visualize_data(mu, correlation, path="plots/", full=True):

    plot_means(mu, labels=ATLAS, path=path)
    plot_covariance(cov=correlation - np.diag(np.diag(correlation)), labels=ATLAS, path=path)
    # Simplified versions of the data 
    df = pd.DataFrame({"regions": ATLAS, "mu": mu})

    if full:
        df[['level', 'matter', 'parcel', 'side']] = df['regions'].str.split('_', expand=True)
        df_levels = df.groupby(['matter', 'parcel'])['mu'].mean().reset_index()
        labels = df_levels.apply(lambda row: "_".join(row[['matter', 'parcel']]), axis=1).values
        plot_means(df_levels['mu'],labels, "_levels", path)


        correlation_df = pd.DataFrame(correlation, index=ATLAS, columns=ATLAS)
        def get_matter_and_parcel(region):
            return "_".join(np.array(region.split('_'))[[1, 2]])
        row_groups = correlation_df.index.map(get_matter_and_parcel)
        correlation_levels = correlation_df.groupby(row_groups).mean().T.groupby(row_groups).mean()

        plot_covariance(cov=correlation_levels - np.diag(np.diag(correlation_levels)), labels=correlation_levels.columns.tolist(), name="_levels", path=path)
    
        white_matter_mapping = {"CST": "I", "FC": "D", "FG": "D", "SL": "V"}
        df['parcel_simple'] = df.apply(lambda region: white_matter_mapping[region['parcel']] if region['matter'] == 'W' else region['parcel'], axis=1)
        df_simple = df.groupby(['level', 'matter', 'parcel_simple'])['mu'].mean().reset_index()
        labels = df_simple.apply(lambda row: "_".join(row[['level', 'matter', 'parcel_simple']]), axis=1).values
        plot_means(df_simple['mu'], labels, "_simple", path)

        def get_simple_map(region): 
            split = np.array(region.split('_'))
            if split[1] == "W": split[2] = white_matter_mapping[split[2]]
            return "_".join(split[[0, 1, 2]])

        correlation_df = pd.DataFrame(correlation, index=ATLAS, columns=ATLAS)
        row_groups = correlation_df.index.map(get_simple_map)
        correlation_simple = correlation_df.groupby(row_groups).mean().T.groupby(row_groups).mean()
        plot_covariance(cov=correlation_simple - np.diag(np.diag(correlation_simple)), labels=correlation_simple.columns.tolist(), name="_simple", path=path)
    

def main():

    parser = argparse.ArgumentParser(description="Run Graph Laplacian Mixture Model")

    parser.add_argument('-p', '--data_path', type=str, help='X data path', required=True)
    parser.add_argument('-simple', '--init_laplacians_pearson', action="store_true", help='initialize laplacians with functional connectivity', default=False)
    args = parser.parse_args()

    X = z_score(np.load(args.data_path)) # design matrix

    mu, sigma, correlation = get_estimators(X)
    visualize_data(mu, correlation)
    print("Done")

if __name__ == '__main__':
    main()

