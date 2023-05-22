# annealed-nce

This repository is dedicated to methods that estimate the normalization of a target distribution,
by contrasting with a proposal distribution.

### Users

First create an environment for this project by running 

```bash
# Create and activate an environment
conda env create -f environment.yml
conda activate annealed-nce
```

Then, install the package

```bash
# Install the package
python setup.py develop
```

<!-- 
### Contributors

First create an environment for this project by running 

```bash
# Create and activate an environment
conda env create -f environment.yml
conda activate annealed-nce
```

Then, install the package, using the editable mode

```bash
# Install the package with editable mode
pip install -e .
```

And use test-driven development to make sure the edits you make do not break unitary tests

```bash
# Run pytest on all files starting with 'test_'
pytest annealednce --cov-report term-missing -v
```
-->

### Run an example

```bash
# Run annealed NCE on different statistical models
ipython -i experiments/run_experiments.py   # interactive mode for debugging when breaks
```


<!-- 
```bash
# Visualize metrics (during or after the experiment)
tensorboard --logdir results
```
-->

### Reference

If you use this code in your project, please cite the following bibtex [to do].
