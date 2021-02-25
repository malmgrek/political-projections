# Dimensionality reduction demo application

## Summary 

This project implements some well-known dimensionality reduction methods 

- Principal component analysis (PCA),
- Independent component analysis (ICA)
- Factorial analysis (FA)

for the "expert level dataset" [2019 Chapel Hill expert
survey](https://www.chesdata.eu/2019-chapel-hill-expert-survey "2019_CHES"). In
principle similar methods are applicable to any (e.g.) standardizable survey
datasets. The main aim is to demonstrate a possible layout of a Python
data analysis project as well as setting up an interactive [Plotly
Dash](https://dash.plotly.com/ "Dash") service for exploring the results.

For opinionated discussion about the methodology, see
[Discussion](doc/discussion.md "Discussion").


<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [User guide](#user-guide)
    - [Quick start](#quick-start)
    - [For developers](#for-developers)
        - [Install development environment](#install-development-environment)
        - [Unit testing](#unit-testing)
        - [Running the Dash service locally](#running-the-dash-service-locally)
        - [Jupyter notebooks and Git](#jupyter-notebooks-and-git)

<!-- markdown-toc end -->


## User guide

### Quick start

Build and run application using Docker:
``` shell
make up
```

Stop the running container podman
``` shell
make stop
```

When the application started succesfully, there should be a Dash application
running at <http://localhost:8050>. Currently, setting up the container has been tested only on
Fedora 32 and Podman instead of Docker.

### For developers

#### Install development environment

Install dependencies and the package using Conda:

``` shell
conda env create --force --name dimred-demo --file environment.yml
source /path/to/conda/bin/activate dimred-demo
pip install -e .
```

Optionally, install manually useful development tools such as Jupyter.

#### Unit testing

``` python
pytest -v          # Run all tests except webtests
pytest --webtest   # Run all tests including webtests
```

#### Running the Dash service locally

The Plotly Dash service can be launched with

``` shell
python app.py
```

By default, the service runs at <http://localhost:8050>.

#### Jupyter notebooks and Git

If you want to track Jupyter notebooks, purge them before staging:

``` shell
bash bin/clean_notebooks
```
