# Provable benefits of annealing for estimating normalizing constants: Importance Sampling, Noise-Contrastive Estimation, and beyond

Code for the paper <a href="" target="_blank">Provable benefits of annealing for estimating normalizing constants: Importance Sampling, Noise-Contrastive Estimation, and beyond</a>. 

### How to install

From within your local repository, run

```bash
# Create and activate an environment
conda env create -f environment.yml
conda activate annealed-nce

# Install the package
python setup.py develop
```

### How to replicate Figures 1, 2, and 3

With current parameters, each script is parallelized over 100 CPUs and takes about 100GB of RAM and a maximum of 7 hours to replicate.

```bash
# Evaluate how the loss, parameter distance, and dimensionality impact the estimation error
ipython -i experiments/01_run_experiment_loss.py
ipython -i experiments/02_run_experiment_distance.py
ipython -i experiments/03_run_experiment_dimension.py

# Plot results
ipython -i experiments/01_plot_experiment_loss.py
ipython -i experiments/02_plot_experiment_distance.py
ipython -i experiments/03_plot_experiment_dimension.py
```

### Reference

If you use this code in your project, please cite:

```bib
@InProceedings{chehab2022annealingnormalizingconstant,
  title = 	 {Provable benefits of annealing for estimating normalizing constants: Importance Sampling, Noise-Contrastive Estimation, and beyond},
  author =       {Chehab, Omar and Hyv{\"a}rinen, Aapo and Risteski, Andrej},
  booktitle = 	 {Neural Information Processing Systems (NeurIPS)},
  year = 	 {2023},
}
```

