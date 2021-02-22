# Dimensionality reduction demo


## Introduction

This project implements some simple dimensionality reduction methods for the
"expert leven dataset" [2019 Chapel Hill expert
survey](https://www.chesdata.eu/2019-chapel-hill-expert-survey "2019_CHES"). In
principle similar methods are applicable to any (e.g.) standardizable survey
datasets. However, the main aim is to demonstrate a possible layout of a Python
data analysis project.

## Instructions

### Jupyter notebooks and Git
To avoid messing up Git history with Jupyter notebook side-effects, notebooks
should be cleaned before committing:

``` shell
bash bin/clean_notebooks
```

### Unit tests

Run (offline) tests from the repo base directory:

``` python
pytest -v
```

The webtests for data loading can be included with 

``` python
pytest --webtest
```

