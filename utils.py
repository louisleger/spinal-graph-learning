import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from graph_learn.clustering.glmm import GLMM

def plot_laplacians(model):
    k = len(model.weights_)
    fig, ax = plt.subplots(2, k, figsize = (k*5 , 10))
    for idx in range(k):
        ax[0, idx].imshow(model.laplacians_[idx])
        adjacency = np.diag(model.laplacians_[idx].diagonal()) - model.laplacians_[idx]

        ax[1, idx].imshow(adjacency)
        ax[0, idx].set_title(f'Laplacian K={idx}')
        ax[1, idx].set_title(f'Adjacency K={idx}')
    plt.tight_layout()
    plt.show()

def plot_adj(model):
    k = len(model.weights_)
    fig, ax = plt.subplots(1, k, figsize = (k*5 , 10))
    for idx in range(k):
        adjacency = np.diag(model.laplacians_[idx].diagonal()) - model.laplacians_[idx]
        ax[idx].imshow(adjacency)
        ax[idx].set_title(f'Adjacency K={idx}')
    plt.tight_layout()
    plt.show()

def identity(x): return x

def z_score(x):
    mean, std = np.mean(x),np.std(x)
    return (x-mean)/(std + 1e-4)

def add_labels_to_ax(ax, labels):
    ax.set_xticks(np.arange(len(labels))) 
    ax.set_yticks(np.arange(len(labels)))  
    ax.set_xticklabels(labels, rotation=90)  
    ax.set_yticklabels(labels)
    return ax

from sklearn.metrics import silhouette_score
def plot_probability_estimates(gamma, timings, save_path="plots/latest_run_probabilties.png", gamma_std=None):
    dt = np.arange(0, len(gamma))


    fig, ax = plt.subplots(gamma.shape[-1], 1, figsize = (10, 3*gamma.shape[-1]), sharex=True, sharey=True)
    for i in range(len(ax)):
        ax[i].fill_between(dt, y1=0, y2=1, color = 'yellow', alpha=0.3,
                        where = timings == True, label="Wrist Movement")
        ax[i].fill_between(dt, y1=0, y2=1, color = 'blue', alpha=0.3, where = timings == False, label="Rest")
        ax[i].plot(gamma[:, i], color='k')
        if gamma_std is not None: 
            ax[i].fill_between(dt, y1=gamma[:, i] - gamma_std[:, i], y2=gamma[:, i] + gamma_std[:, i], color='k', alpha=0.1)
        ax[i].set_ylabel(f'K={i}', rotation = 0, labelpad = 20, fontsize = 12)

    fig.suptitle('Probability Estimates $\gamma$', fontsize = 16)
    ax[-1].set_xlabel('$\Delta t$ \n(TR = 2.5s)', fontsize = 12)
    ax[-1].set_ylim(0, 1)
    ax[-1].set_xlim(0, dt.shape[-1] - 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")

def cluster_score(gamma, activity, data, prediction):
    gamma = gamma > gamma.mean(axis=0)
    if (len(activity.shape) == 1): activity = activity.reshape(-1, 1)
    tp = np.sum((gamma == 1) & (activity == 1), axis=0)
    fp = np.sum((gamma == 1) & (activity == 0), axis=0)
    fn = np.sum((gamma == 0) & (activity == 1), axis=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = np.round(2 * (precision * recall) / (precision + recall + 1e-8), 3)
    if len(np.unique(prediction)) == 1: return (f1_score, np.nan)
    return (f1_score, silhouette_score(data, prediction))