# annealed-nce

This repository is dedicated to methods that estimate the log-normalization of a target distribution,
by contrasting with a proposal distribution.

### Users

First create an environment for this project by running 

```bash
# Create and activate an environment
conda env create -f environment.yml
conda activate annealed-nce
```

Then, install the package. For users, proceed with a regular installation

```bash
# for users: install the package
python setup.py develop
```

For contributors, install the package using the editable mode

```bash
# for contributors: install the package using the editable mode
pip install -e .
```

### Run an example

```bash
# Run an experiment
ipython -i experiments/01_run_experiment_loss.py

# Plot results 
ipython -i experiments/01_plot_experiment_loss.py
```

