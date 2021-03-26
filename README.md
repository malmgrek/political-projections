# Dimensionality reduction demo application

## Summary 

This project implements some well-known dimensionality reduction methods 

- Principal component analysis (PCA),
- Independent component analysis (ICA)
- Factor analysis (FA)

for the [Finnish national broadcasting company
survey](https://yle.fi/uutiset/3-10725384 "2019 YLE") for parliamentary
electoral candidates, as well as the "expert level dataset" in the [2019 Chapel
Hill expert survey](https://www.chesdata.eu/2019-chapel-hill-expert-survey
"2019_CHES"). In principle, analogous methodology is applicable to any
standardizable survey datasets. The main aim of this project to demo how an
interactive [Plotly Dash](https://dash.plotly.com/ "Dash") service works for
exploring the results. Nevertheless, closer examination of the results may
reveal some interesting facts about present day politics such as clustering of
typical questions regarding nationalism, multiculturalism, liberalism,
left/right and so on.

A version of the app is running at <https://political-projections.herokuapp.com>.


<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [User guide](#user-guide)
    - [Quick start](#quick-start)
    - [For developers](#for-developers)
        - [Install development environment](#install-development-environment)
        - [Unit testing](#unit-testing)
        - [Running the Dash service locally](#running-the-dash-service-locally)

<!-- markdown-toc end -->


## User guide

### Quick start

Build and run application locally using Docker:
``` shell
make up
```

Stop the running container:
``` shell
make stop
```

When the application started succesfully, there should be a Dash application
running at <http://localhost:8050>. 

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
