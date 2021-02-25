# Chapel Hill 2019 – discussion of results

The goal of the project was to implement a 2D dimensionality reduction for the
political parties scores in the[2019 Chapel Hill expert
survey](https://www.chesdata.eu/2019-chapel-hill-expert-survey "CHES2019")
dataset, as well as a model that can be used to "random generate" hypothetical
political parties based on their 2D distribution (w.r.t. the dimension reduction
method). Principal component analysis (PCA) together with data scaling and
feature clustering/selection was studied the most but some experiments with
Independent component analysis (ICA) and Factor analysis (FA) were also done,
and the methods are included in the Dash dashboard. Below we only discuss the
PCA related findings / methodology.

## Key findings

PCA indicates that most significant spread in the dataset's parties' views lie
along the _Liberal/multiculturalism/ecologism_ –
_Nationalism/conservativism/anti-immigration_ axis. Secondly, but less
significantly, the _Economic left_ – _right_ axis also importantly explains variation in
the questionnaire responses. The "ultra-right" seems to form a separate
cluster which is at a very extreme on the first axis but right-centrist on the
other axis. All other type of parties form a continuum blob.


<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Analysis pipeline](#analysis-pipeline)
    - [1. Data preparation](#1-data-preparation)
        - [1.1 Manual feature overview](#11-manual-feature-overview)
        - [1.2 Clean up data](#12-clean-up-data)
        - [1.3 Scale features and optionally standardize](#13-scale-features-and-optionally-standardize)
        - [Re-order features based on hierarchicaly clustering](#re-order-features-based-on-hierarchicaly-clustering)
    - [2. Dimensionality reduction](#2-dimensionality-reduction)
        - [2.1 Min/max bounds reduced to 2D](#21-minmax-bounds-reduced-to-2d)
    - [3. Kernel density estimation in 2D](#3-kernel-density-estimation-in-2d)
        - [3.1 Transform back to original dimensions](#31-transform-back-to-original-dimensions)
- [Applications](#applications)

<!-- markdown-toc end -->


## Analysis pipeline

### 1. Data preparation

#### 1.1 Manual feature overview

Most of the survey questions are consider the political parties but some
consider

- metadata such as party nomenclature and respondent date of birth,
- so called "vignette" questions which can be used to benchmark how biased
  the respondents are.
  
Nevertheless, the dropped questions are likely to contain interesting
information regarding geographic distribution and normalization.

> **Possible improvements:** Use the vignette questions to remove respondent
> biases from the data. Could perhaps be done using e.g. linear regression.

#### 1.2 Clean up data

1. Filter NaN's and transform to numeric data type.
2. Group by party identifiers with median (or mean). 

Point 1 is just a routine action but 2 is very important because we primarily
want to study the _parties_ and not the _respondents views_. If we didn't group
the data as above, we would be solving the wrong problem. Moreover, some parties
with most evaluations are from quite small countries such as Croatia or Czech
republic so the raw data based analysis might be biased towards country specific
situation.

> **Possible improvements:** Not all parties' observations are equally "noisy".
> Some are much more frequent than others, and group (same party, different
> respondanr) variances vary. Hence a noise adaptive method such as weighted PCA
> would make sense. Left for future studies.

#### 1.3 Scale features and optionally standardize

The Chapel Hill documentation shows the question specific scales -- the scales
are 1–10, 0–10, 1–7, ..., that is, the data need to be scaled to uniform units.
Two unit transformations were experimented:

1. Affine transformation to the interval [-1, 1] 
2. Affine transformation to zero mean and unit variance (standardization)

Method 1 is useful if we want to preserve the information on varying spreads
among features. For example, immigration views divide people more than views on
culture or regional policymaking. Method 2 is useful if we want to study
correlation, not covariance.

#### Re-order features based on hierarchicaly clustering

It is not hard to interpret the principal components information content if the
features are in random order. I ran a hierarchical clustering & dendrogram that
group together features whose correlations with other features are similar as
vectors. With this done, it is interesting to look at the correlation/covariance
matrix as image and compare it with the principal components. It also enables
grouping features if desired.

> **Possible extensions:** Drop out / group features that correlate strongly to
> give more interpretability to PCA results.

### 2. Dimensionality reduction

Scikit-learn provides an out of the box PCA object although calculating a
rudimentary (non-probabilistic) PCA is straightforward: it is characterized
by the eigendecomposition of the covariance matrix. Dimensionality reduction can
be calculated by (centering +) projecting the original N-dimensional vectors to
a couple of first eigenvectors.

#### 2.1 Min/max bounds reduced to 2D

The min/max bounds are orthogonal planes in the N-dimensional space so the
survey region is a cuboid. It is enough to find the intersection of each
transformed plane with the "`xy`-plane" defined by the first two principal
components. This can be solved analytically (see comments in `analysis.py`) by
expressing a plane in form `normal . x = a . x`. Since the PCA transformation is
`Translation + Rotation`, its `normal` is just rotated whereas `a` is projected
onto the line spanned by `normal`, and rotated.

> **Remark:** Some of the _projected_ data points may be out of the _projected_ bounds in 2D because the remaining dimensions are ignored.

### 3. Kernel density estimation in 2D

To random draw "new political parties", I fitted a kernel density function over
the 2 projected point cloud, and used grid search to fine tune the smoothness
parameter. Scikit-learn's KDE tool is nice as it provides also a method for
random sampling the fitted distribution. The density fit itself is a linear
combination of Gaussian bells, so it can unfortunately produce random samples
outside of the original hard limits.

#### 3.1 Transform back to original dimensions

Points in the reduced dimension can be transformed to the original dimensions
with pseudoinverse that maps to the minimum norm solution. Unit transformation
is a one-to-one mapping. To ensure "physical" outcomes in original coordinates,
we clip to min/max limits and round to nearest integer. This is not exactly
accurate but will do for now.

> **Possible improvements:** Mathematically rigorous way of restricting the probability distribution.

## Applications

I implemented a semi-realistic web application that can be used for visualizing
and studying the analysis results. In fact, considering the survey data size and the fact that it changes rarely, the application is very lightweight on back-end.
Back-end processing load can still be reduced to very low level through
memoization.

If the service had a million or so users per hour, the main bottleneck would be
the memory / processing load on back-end server. I would consider using
Kubernetes for running a cluster of containerized applications, and a load
balancer for directing the network requests efficiently among the containers in
the cluster. Such a system would scale up/down appropriately with respect to
load, and the maintenance cost would be proportional to cloud resource usage.
