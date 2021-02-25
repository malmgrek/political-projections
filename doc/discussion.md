# Chapel Hill 2019 – discussion of results

The goal of the project was to implement a 2D dimensionality reduction for the
political parties scores in the Chapell Hill 2019 dataset. Furthermore, a model
that can be used to "random generate" hypothetical political parties based on
their 2D distribution (w.r.t. the dimension reduction method). Principal
component analysis (PCA) together with data scaling and feature
clustering/selection was studied the most but some experiments with Independent
component analysis (ICA) and Factorial analysis (FA) were also done, and the
methods are included in the Dash dashboard. Below we only discuss the PCA
related findings.

## Key findings

- First component: Anti-immigration + nationalist + Authoritarian + conservative <-> Liberal
- Second component: Traditional Left / Right
- Extreme right seems to form a separate cluster in 2D


<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Analysis pipeline](#analysis-pipeline)
    - [1. Data preparation](#1-data-preparation)
        - [1.1 Manual feature overview](#11-manual-feature-overview)
        - [1.2 Clean up data](#12-clean-up-data)
        - [1.3 Scale features and optionally standardize](#13-scale-features-and-optionally-standardize)
        - [Re-order features based on hierarchicaly clustering](#re-order-features-based-on-hierarchicaly-clustering)
    - [2. Dimensionality reduction](#2-dimensionality-reduction)
        - [2.1 Questions' hard limits in 2D](#21-questions-hard-limits-in-2d)
    - [3. Kernel density estimation in 2D](#3-kernel-density-estimation-in-2d)
    - [4. Transform back to original dimensions](#4-transform-back-to-original-dimensions)
- [Applications](#applications)

<!-- markdown-toc end -->


## Analysis pipeline

### 1. Data preparation

#### 1.1 Manual feature overview

Most of the questions are directly related to the characterizing the parties but
some were

- metadata such as party nomenclature and respondent date of birth,
- so called "vignette" questions which can be used to benchmark how biased
  the respondents are.
  
Such features were dropped out from this analysis. Obviously the approach
doesn't scale well but it's important to try to understand the data if possible.

> **Possible improvements:** Use the vignette questions to remove respondent biases from the data. Could perhaps be done using e.g. linear regression.

#### 1.2 Clean up data

1. Filter NaN's and transform to numeric data type.
2. Group by party identifiers with median (or mean). 

Point 1 is just a routine action but 2 is very important. The point is we
primarily want to study the parties and not the respondents views. If we didn't
group the data as above, we would be solving the wrong problem. Moreover, some
parties with most evaluations are from quite small countries such as Croatia or
Czech republic so the raw data based analysis would be biased towards them.

> **Possible improvements:** Not all parties' observations are equally "noisy". Some are much more frequent than others, and group variances vary. Hence a noise adaptive method such as weighted PCA would make sense. Left for future studies.

#### 1.3 Scale features and optionally standardize

The Chapel Hill documents tell the question specific scales -- the scales are 1–10, 0–10, 1–7, ..., that is, the data need to be scaled. Two unit transformations were experimented:

A) Affine transformation to the interval [-1, 1] 
B) Affine transformation to zero mean and unit variance (standardization)

Method A is useful if we want to preserve the information on varying spreads
among features. For example, immigration views divide perhaps more than views on
religious freedom. Method A is useful if we want to study correlation, not
covariance.

#### Re-order features based on hierarchicaly clustering

It is not easy to interpret the principal components information content if the
features are in random order. Thus I ran a hierarchical clustering + dendrograph
that groups together features whose correlations with other features are similar
as vectors. With this done, it is interesting to look at the
correlation/covariance matrix as image and compare it with the principal
components. It also enables grouping features if desired.

### 2. Dimensionality reduction

Scikit-learn provides an out of the box PCA object although calculating a
rudimentary (non-probabilistic) PCA is very straightforward: it is characterized
by the eigendecomposition of the covariance matrix. Dimensionality reduction can
be calculated by (centering +) projecting the original N-dimensional vectors to
a couple of first eigenvectors.

#### 2.1 Questions' hard limits in 2D

The hard limits are actually orthogonal planes in the N-dimensional space so the
survey region is a cuboid. It is enough to find the intersection of each plane
with the "`xy`-plane" defined by the first two principal components. This can be
solved analytically (see `analysis.py`) by expressing a plane in form `normal .
x = a . x`. Since the PCA transformation is `Translation + Rotation`, its
`normal` is just rotated whereas `a` is projected onto the line spanned by
`normal`, and rotated.

> **Remark:** Some of the _projected_ data points may be out of the _projected_ bounds in 2D because the remaining dimensions are ignored.

### 3. Kernel density estimation in 2D

To random draw "new political parties", I fitted a kernel density function over
the 2 projected point cloud using grid search to fine tune smoothness parameter.
Scikit-learn's KDE tool is good as it provides also an API for random sampling
from the fitted distribution. The fit is a linear combination of bell surfaces,
so it can produce random samples outside of the original hard limits.

> **Possible improvements:** Mathematically rigorous way of restricting the probability distribution.

### 4. Transform back to original dimensions

Points in the reduced dimension can be transformed to the original dimensions
with pseudoinverse that maps to the minimum norm solution. Unit transformation
is a one-to-one mapping. To ensure "physical" outcomes in original coordinates,
we clip to min/max limits and round to nearest integer. This is not exactly
accurate but will do for now.

## Applications

For demonstration purposes, I wanted to implement a semi-realistic web
application that could be used for visualizing results. In fact, considering the
survey data size and unchanging nature, the application is light weight on
back-end. Due to the light-weight nature of the methodology, back-end processing
resource usage can be reduced to low level through memoization. 

If the service would have million or so users per hour, the main bottleneck
would be the data traffic load on back-end server as well as network. I would
consider using e.g. the Kubernetes Engine to serve a containerized application
in a dynamically scaling cluster and Ingress for load balancing the network
traffic.

<!-- Millions users: load balancing, -->
<!-- multiple machines, Kubernetes -->

<!-- - Load balancing: -->
<!--   reverse proxy distributes the requests -->
<!-- - defines resources before hand -->
<!-- - master node -> worker nodes -->
<!-- - in principle, scales w.r.t. load -->
<!-- - does the master proxy know loads of subjects? -->
<!-- - Google Kubernetes Engine out of the box solution -->


<!-- ## Dimensionality reduction of Chapel Hill 2019 dataset -->

<!-- The Chapel Hill dataset -->
<!-- [CHESDATA](https://www.chesdata.eu/2019-chapel-hill-expert-survey) contains -->
<!-- survey data regarding different political parties in EU member states. The -->
<!-- respondents are political science experts, and the questions cover a wide range -->
<!-- of topics related to views about economy, social policy, culture, immigration -->
<!-- and so on. -->

<!-- ### Remarks on the dataset -->

<!-- - The survey has been completed by political scientists, and not directly by party -->
<!-- members. -->
<!-- - Some columns are unrelated to -->

<!-- ### Methodology -->
<!-- - Data always centered -->
<!-- - Standardization not a good idea if the aim is to be able to produce an -->
<!--   interesting set of "random parties". Otherwise the difference in variances -->
<!--   will be hidden although it is interesting. -->

<!-- ### Notes -->

<!-- - Because of curse of dimensionality, it is hard to model density in higher dimensions -->
<!-- - It is hard to sample in higher dimensions -->
<!-- - In far-right--liberal axis most variation in absolute terms -->
<!-- - In traditional left-right axis most correlation -->
<!-- - Span of smaller components can be added to samples and they will still inverse transform -->
<!--   to same values -->
<!-- - When projected, some original data points fall outside of the polygon. This as expected, -->
<!--   as the projection ignores the rest of the PCA coordinates. In particular, if the other coordinates -->
<!--   are used, the points will of course inverse map back to the bounds. If we set other components to 0, -->
<!--   then the projected points will map outside of the bounds! -->
<!-- - Separate md file with answers to questions -->
<!-- - Run full pipeline: Clone repo -> make podman-up -> use app -->
