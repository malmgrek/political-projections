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

Install the package and its dependencies 

Using Conda:

``` shell
conda env create --force --name dimred-dev --file environment.yml
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
