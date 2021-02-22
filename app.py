"""Plotly Dash app for displaying analysis results

TODO:
- CB: NaN filtering parameters
- Map samples to the integer grid
- Plot principal values
- Train/test split validation


"""

import random

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from flask_caching import Cache
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn import decomposition, preprocessing

from dimred import analysis
from dimred.datasets import ches2019


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": "cache-directory",
        # Keep only max 5 cache files
        "CACHE_THRESHOLD": 5,
    }
)


TIMEOUT = 10


def checklist_to_bool(x):
    return x is not None and "v" in x


def IntervalScaler(features):
    return analysis.IntervalScaler([
        v for (k, v) in ches2019.feature_scales.items() if k in features
    ])


def reorder_features(X, features, corrcov):
    #
    # TODO/FIXME: This is an ugly workaround
    # for ordering the features as in the dendrogram
    #
    C = np.cov(X.T) if corrcov == "cov" else spearmanr(X).correlation
    fig = ff.create_dendrogram(
        C,
        orientation="bottom",
        labels=features,
        linkagefun=hierarchy.ward
    )
    new_features = fig["layout"]["xaxis"]["ticktext"]
    x = pd.DataFrame(X, columns=features)
    x = x[new_features]
    return (x.values, new_features)


@cache.memoize(timeout=TIMEOUT)
def get_training_data():
    x = ches2019.download()
    # x = ches2019.load()
    x = ches2019.cleanup(x, nan_floor_row=0.9, nan_floor_col=0.75)
    (X, features) = ches2019.prepare(x)
    return pd.DataFrame(X, columns=features).to_json(orient="split")


def Dataset(meanvar=None, impute=None, corrcov=None):

    meanvar_bool = checklist_to_bool(meanvar)
    impute_bool = checklist_to_bool(impute)

    training_data = pd.read_json(get_training_data(), orient="split")

    # Optionally impute
    X = (
        analysis.impute(training_data.values, max_iter=21) if impute_bool
        else training_data.dropna().values
    )
    features = training_data.columns

    scaler = (
        analysis.StandardScaler().fit(X) if meanvar_bool else
        IntervalScaler(features)
    )

    # Scale
    X = scaler.transform(X)
    (X, features) = reorder_features(X, features, corrcov)

    return (X, features, scaler)


app.layout = html.Div([
    html.H2(
        "Chapel Hill expert survey"
    ),
    html.P(
        "Dimensionality reduction to two dimensions with various methods"
    ),
    html.A(
        html.P(
            "Code book"
        ),
        href="https://www.chesdata.eu/s/2019_CHES_codebook.pdf",
        target="_blank"
    ),
    html.Div(
        children=[
            dcc.Dropdown(
                id="dropdown-method",
                options=[
                    {"label": "Principal component analysis", "value": "pca"},
                    {"label": "Independent component analysis", "value": "ica"},
                    {"label": "Rotated factiorial analysis", "value": "fa"}
                ],
                value="pca",
                style={"margin-top": "2em", "width": "20em"}
            ),
            dcc.Dropdown(
                id="dropdown-corrcov",
                options=[
                    {"label": "Covariance in heatmap", "value": "cov"},
                    {"label": "Spearman rank in heatmap", "value": "corr"},
                ],
                value="cov",
                style={"margin-top": "1em", "width": "20em"}
            ),
            dcc.Input(
                id="input-components",
                type="number",
                placeholder="Number of components visualized",
                min=1,
                max=1000,
                value=3,
                style={"margin-top": "1em", "width": "20em"}
            ),
            dcc.Checklist(
                id="checklist-meanvar",
                options=[
                    {"label": "Map to zero mean and unit variance", "value": "v"},
                ],
                style={"margin-top": "1em", "width": "20em"}
            ),
            dcc.Checklist(
                id="checklist-impute",
                options=[
                    {"label": "Impute NaNs (otherwise drop)", "value": "v"},
                ],
                style={"margin-top": "1em", "width": "20em"}
            ),
            dcc.Checklist(
                id="checklist-whiten",
                options=[
                    {"label": "Whiten", "value": "v"},
                ],
                style={"margin-top": "1em", "width": "20em"}
            ),
        ]
    ),
    dcc.Loading(
        children=dcc.Graph(id="graph-training-heatmap"),
    ),
    dcc.Loading(
        children=dcc.Graph(id="graph-corr-heatmap"),
    ),
    dcc.Loading(
        children=dcc.Graph(id="graph-components"),
    )
], style={"width": "800px", "margin": "0 auto"})


@app.callback(
    Output("graph-training-heatmap", "figure"),
    [
        Input("dropdown-method", "value"),
        Input("checklist-meanvar", "value"),
        Input("checklist-impute", "value"),
        Input("dropdown-corrcov", "value"),
    ]
)
def update_training_heatmap(method, meanvar, impute, corrcov):

    (X, features, scaler) = Dataset(meanvar, impute, corrcov)

    # Plot dataset
    fig = px.imshow(
        X,
        x=features,
        height=800,
        width=800,
        color_continuous_scale="Agsunset",
        title="Processed training data, shape: {}".format(X.shape)
    )

    return fig


@app.callback(
    Output("graph-corr-heatmap", "figure"),
    [
        Input("dropdown-method", "value"),
        Input("checklist-meanvar", "value"),
        Input("checklist-impute", "value"),
        Input("dropdown-corrcov", "value"),
    ]
)
def update_corr_heatmap(method, meanvar, impute, corrcov):

    (X, features, scaler) = Dataset(meanvar, impute, corrcov)
    C = np.cov(X.T) if corrcov == "cov" else spearmanr(X).correlation
    # corr_linkage = hierarchy.ward(corr)

    fig = ff.create_dendrogram(
        C,
        orientation="bottom",
        labels=features,
        linkagefun=hierarchy.ward
    )

    for i in range(len(fig["data"])):
        fig["data"][i]["yaxis"] = "y2"

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(
        C,
        orientation="right",
        labels=features,
        linkagefun=hierarchy.ward
    )
    for i in range(len(dendro_side["data"])):
        dendro_side["data"][i]["xaxis"] = "x2"

    # Add Side Dendrogram Data to Figure
    for data in dendro_side["data"]:
        fig.add_trace(data)

    # Create Heatmap
    dendro_leaves = dendro_side["layout"]["yaxis"]["ticktext"]

    dendro_leaves = list(range(len(dendro_leaves)))
    heat_data = C
    heat_data = heat_data[dendro_leaves,:]
    heat_data = heat_data[:,dendro_leaves]

    heatmap = [
        go.Heatmap(
            x=dendro_leaves,
            y=dendro_leaves,
            z=heat_data,
            colorscale="Agsunset"
        )
    ]

    heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
    heatmap[0]["y"] = dendro_side["layout"]["yaxis"]["tickvals"]

    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)

    # Edit Layout
    fig.update_layout({
        "width": 800,
        "height": 800,
        "showlegend": False,
        "hovermode": "closest",
        "template": "plotly_white"
    })
    # Edit xaxis
    fig.update_layout(
        xaxis={
            "domain": [.15, 1],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "ticks": ""
        }
    )

    for (key, domain) in zip(
            ["xaxis2", "yaxis", "yaxis2"],
            [[0, .15], [0, .85], [.825, .975]]
    ):
        fig.update_layout(**{
            key: {
                "domain": domain,
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": False,
                "ticks": ""
            }
        })

    return fig


@app.callback(
    Output("graph-components", "figure"),
    [
        Input("dropdown-method", "value"),
        Input("checklist-meanvar", "value"),
        Input("checklist-impute", "value"),
        Input("checklist-whiten", "value"),
        Input("dropdown-corrcov", "value"),
        Input("input-components", "value"),
    ]
)
def update_components(method, meanvar, impute, whiten, corrcov, components):

    whiten_bool = checklist_to_bool(whiten)

    (X, features, scaler) = Dataset(meanvar, impute, corrcov)

    # Form decomposition
    if method == "pca":
        decomposer = decomposition.PCA(whiten=whiten_bool).fit(X)
    if method == "ica":
        decomposer = decomposition.FastICA(
            random_state=np.random.RandomState(42),
            whiten=whiten_bool,
            max_iter=2000
        ).fit(X)
    if method == "fa":
        decomposer = decomposition.FactorAnalysis(rotation="varimax").fit(X)
    Y = decomposer.transform(X)
    Y_2d = Y[:, :2]
    U = decomposer.components_
    # V = U
    V = scaler.inverse_transform(U)

    # Fit KDE and sample FIXME: Fix sampling to limits using Numpy hack
    kde = analysis.fit_kde(Y_2d)
    num_samples = 10
    Y_samples = kde.sample(num_samples)
    (x, y, density, xlim, ylim) = analysis.score_density_grid(
        kde=kde, Y=Y_2d, num=100
    )
    # Inverse transform not supported for FA :(
    Y_samples_full = np.hstack((
        Y_samples,
        np.zeros((num_samples, len(features) - 2))
    ))
    if method in ["pca", "ica"]:
        X_samples = scaler.inverse_transform(
            decomposer.inverse_transform(Y_samples_full)
        )
        # X_samples = decomposer.inverse_transform(Y_samples_full)

    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [
                {"rowspan": 1, "colspan": 2},
                None
            ],
            [
                {"rowspan": 1, "colspan": 2},
                None
            ],
            [
                {"rowspan": 1, "colspan": 1, "type": "surface"},
                {"rowspan": 1, "colspan": 1}
            ],
            [
                {"rowspan": 1, "colspan": 2},
                None
            ],
        ],
        subplot_titles=[
            "Components in normalized coordinates",
            "Components (in orignal coordinates)" + (
                ", explained variance: {0:.0%}".format(
                    decomposer.explained_variance_ratio_[:components].sum()
                ) if method == "pca" else ""
            ),
            "Probability density",
            "Samples in 2D",
        ] + (
            ["Samples in original coordinates"] if method in ["pca", "ica"] else
            ["Reverse transformation for factorial analysis not supported :("]
        )
    )

    colors = random.sample(px.colors.qualitative.Plotly, components)
    for (i, color) in enumerate(colors):
        fig.append_trace(
            go.Scatter(
                x=features,
                y=U[i],
                name="Component {}".format(i),
                line={"color": color}
            ),
            1, 1
        )
    for (i, color) in enumerate(colors):
        fig.append_trace(
            go.Scatter(
                x=features,
                y=V[i],
                line={"color": color},
                showlegend=False
            ),
            2, 1
        )

    fig.append_trace(
        go.Surface(
            x=x,
            y=y,
            z=density,
            showscale=False,
            colorscale="agsunset"
        ),
        3, 1
    )
    fig.append_trace(
        go.Scatter(
            x=Y_2d[:, 0],
            y=Y_2d[:, 1],
            mode="markers",
            marker={
                "color": "black",
                "size": 12,
                "opacity": 0.5
            },
            name="Projected parties"
        ),
        3, 2
    )
    fig.append_trace(
        go.Contour(
            x=x[0, :],
            y=y[:, 0],
            z=density,
            contours_coloring="lines",
            line_width=2,
            showscale=False,
            colorscale="agsunset"
        ),
        3, 2
    )
    for (i, Y_sample) in enumerate(Y_samples):
        fig.append_trace(
            go.Scatter(
                x=[Y_sample[0]],
                y=[Y_sample[1]],
                mode="markers",
                marker={"size": 18},
                name="Sample {}".format(i)
            ),
            3, 2
        )
    if method in ["pca", "ica"]:
        for X_sample in X_samples:
            fig.append_trace(
                go.Scatter(
                    x=features,
                    y=X_sample,
                    showlegend=False
                ),
                4, 1
            )
    if method == "fa":
        fig.append_trace(
            go.Scatter(),
            4, 1
        )

    fig.update_layout(
        autosize=False,
        width=1000,
        height=1600,
        template="plotly_white"
    )

    return fig


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="8050", debug=True)
