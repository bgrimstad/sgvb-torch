# sgvb-torch
An implementation of Stochastic Gradient Variational Bayes (SGVB) in PyTorch

This code is an excerpt of the code used to train model in the paper:

```
@unpublished{Grimstad2020,
  author = {Grimstad, Bjarne and Hotvedt, Mathilde and Sandnes, Anders T. and Kolbj{\o}rnsen, Odd and Imsland, Lars S.},
  archivePrefix = {arXiv},
  arxivId = {2102.01391},
  title = {{Bayesian Neural Networks for Virtual Flow Metering: An Empirical Study}},
  url = {http://arxiv.org/abs/2102.01391},
  year = {2021}
}
```

## Installation (to run locally)

1. Download and install Anaconda (https://www.anaconda.com/).
2. Create a new conda environment: `conda env create -f environment.yml`. 
This will create a new environment called ttk28 with the packages listed in `environment.yml`. 
3. Activate the new environment: `conda activate sgvb-torch`.

## Examples

1. Train a Bayesian linear model: `examples/linear.py`
2. Approximate a sinusoidal function by a Bayesian neural network: `examples/sinusoidal.py`
3. Approximate a two-dimensional function by a Bayesian neural network: `examples/multidim.py`

