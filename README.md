# Dimensionality reduction demo


<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Introduction](#introduction)
- [Code instructions](#code-instructions)
    - [Quick start](#quick-start)
    - [Information for developers](#information-for-developers)
        - [Install development environment](#install-development-environment)
        - [Unit tests](#unit-tests)
        - [Jupyter notebooks and Git](#jupyter-notebooks-and-git)

<!-- markdown-toc end -->



## Introduction

This project implements some simple dimensionality reduction methods for the
"expert leven dataset" [2019 Chapel Hill expert
survey](https://www.chesdata.eu/2019-chapel-hill-expert-survey "2019_CHES"). In
principle similar methods are applicable to any (e.g.) standardizable survey
datasets. However, the main aim is to demonstrate a possible layout of a Python
data analysis project.

## Dimensionality reduction of Chapel Hill 2019 dataset

The Chapel Hill dataset
[CHESDATA](https://www.chesdata.eu/2019-chapel-hill-expert-survey) contains
survey data regarding different political parties in EU member states.

### Remarks on the dataset

- The survey has been completed by political scientists, and not directly by party
members.
- Some columns are unrelated to

### Methodology
- Data always centered
- Standardization not a good idea if the aim is to be able to produce an
  interesting set of "random parties". Otherwise the difference in variances
  will be hidden although it is interesting.

### Notes

- Because of curse of dimensionality, it is hard to model density in higher dimensions
- It is hard to sample in higher dimensions
- In far-right--liberal axis most variation in absolute terms
- In traditional left-right axis most correlation
- Span of smaller components can be added to samples and they will still inverse transform
  to same values
- When projected, some original data points fall outside of the polygon. This as expected,
  as the projection ignores the rest of the PCA coordinates. In particular, if the other coordinates
  are used, the points will of course inverse map back to the bounds. If we set other components to 0,
  then the projected points will map outside of the bounds!
- Separate md file with answers to questions



## Code instructions

### Quick start

Build and run the Dash application from a container:

``` shell
make up    # Build and run application using Docker
make stop  # Stop the running container
```

When the application started succesfully, there should be a [Plotly
Dash](https://dash.plotly.com/ "Dash") application running at
<http://localhost:8050>.


### Information for developers

#### Install development environment

Install dependencies and the package using Conda:

``` shell
conda env create --force --name dimred-demo --file environment.yml
source /path/to/conda/bin/activate dimred-demo
pip install -e .
```

#### Unit tests

``` python
pytest -v          # Run all tests except webtests
pytest --webtest   # Run all tests including webtests
```

#### Jupyter notebooks and Git
If you want to track Jupyter notebooks, purge them before staging:

``` shell
bash bin/clean_notebooks
```
