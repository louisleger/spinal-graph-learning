# Graph Learning of the Spinal Chord

Code repository to reproduce experiments ran for the semester project on unsupervised learning of spinal cord functional connectomes by me. Supervised by Ilaria Ricchi in Dimitri Van de Ville's Lab. 

```
.
├── data
│   ├── fmri
│   ├── resources
│   └── runs
├── lib
│   ├── __init__.py
│   ├── clustering
│   ├── ...
├── plots
│   ├── ...
├── run_glmm.py
├── glmm_gridsearch.py
├── run_synthetic.py
├── synthetic_gridsearch.py
├── main.ipynb
├── synthetic_data.ipynb
├── sdemo.mp4
├── utils.py
├── graph_vis.py
└── visualize_data.py
```

This repository contains full environment I used to create plots in report and presentation. The data is included in data/fmri, resources/ contains the activity paradigm and runs/ contains some outputs of scripts. 

## Real Data Spinal Cord GLMM
run_glmm.py and glmm_gridsearch.py allow you to run a GLMM on specified data path of a .npy matrix shape (n, d).
Example command of run_glmm.py: 
```
run_glmm.py -p data/fmri/spinal_data.npy -k 6 -ag 44 -d 2 --full True
```
Where arguments are shortcuts but you can also write average_degree and delta.

## Synthetic Data Generations

run_synthetic.py allows you to run a sanity check on GLMM for specified data (see synthetic_data notebook for details) (also saves synthetic data means and cov plots).

synthetic_gridsearch.py allows you to run a large data generation simulation to extract a feasability matrix and for the moment just save the results as a plot in plots/

## Misc
utils.py visualize_data.py are just importable modules and lib/ is the graph-learn library locally cloned from https://github.com/LTS4/graph-learning/


