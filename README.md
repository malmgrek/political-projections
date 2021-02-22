# Dimensionality reduction demo


## Introduction

This project implements some simple dimensionality reduction methods for the
"expert leven dataset" [2019 Chapel Hill expert
survey](https://www.chesdata.eu/2019-chapel-hill-expert-survey "2019_CHES"). In
principle similar methods are applicable to any (e.g.) standardizable survey
datasets. However, the main aim is to demonstrate a possible layout of a Python
data analysis project.

## Instructions

### Quick start

Build and run the Dash application from a container:

``` shell
make up    # Build and run application using Docker
make stop  # Stop the running container
```

When the application started succesfully, there should be a [Plotly
Dash](https://dash.plotly.com/ "Dash") application running at
<http://localhost:8050>.


### Developer instructions

### Jupyter notebooks and Git
To avoid messing up Git history with Jupyter notebook side-effects, notebooks
should be cleaned before committing:

``` shell
bash bin/clean_notebooks
```

### Unit tests

``` python
pytest -v          # Run all tests except webtests
pytest --webtest   # Run all tests including webtests
```

