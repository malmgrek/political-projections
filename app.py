"""Plotly Dash app for displaying analysis results

TODO:
- Train/test split validation

"""

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


def IntervalScaler(features):
    return analysis.IntervalScaler([
        v for (k, v) in ches2019.feature_scales.items() if k in features
    ])


def reorder_features(X, features):
    #
    # TODO/FIXME: This is an ugly workaround
    # for ordering the features as in the dendrogram
    #
    corr = spearmanr(X).correlation
    fig = ff.create_dendrogram(
        corr,
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
    x = x.dropna()
    (X, features) = ches2019.prepare(x)
    X = analysis.impute(X, max_iter=21)
    return pd.DataFrame(X, columns=features).to_json(orient="split")


def Dataset(whiten=None, prune_correlated=None):
    whiten_bool = not (whiten is None or "v" not in whiten)
    # whiten = False if whiten is None else True
    training_data = pd.read_json(get_training_data(), orient="split")
    X = training_data.values
    features = training_data.columns
    scaler = (
        IntervalScaler(features) if not whiten_bool else
        preprocessing.StandardScaler().fit(X)
    )
    X = scaler.transform(X)
    (X, features) = reorder_features(X, features)
    if prune_correlated is not None:
        pass
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
            dcc.Checklist(
                id="checklist-whiten",
                options=[
                    {"label": "Whitened data", "value": "v"},
                ],
                style={"margin-top": "1em", "width": "20em"}
            ),
            dcc.Checklist(
                id="checklist-corr",
                options=[
                    {"label": "Prune correlated features", "value": "v"},
                ],
                style={"margin-top": "1em", "width": "20em"}
            ),
            dcc.Checklist(
                id="checklist-dropna",
                options=[
                    {"label": "Drop NaNs", "value": "v"},
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
        Input("checklist-whiten", "value")
    ]
)
def update_training_heatmap(method, whiten):

    (X, features, scaler) = Dataset(whiten)

    # Plot dataset
    fig = px.imshow(
        X,
        x=features,
        height=800,
        width=800,
        color_continuous_scale="Agsunset",
        title="Processed training data"
    )

    return fig


@app.callback(
    Output("graph-corr-heatmap", "figure"),
    [
        Input("dropdown-method", "value"),
        Input("checklist-whiten", "value")
    ]
)
def update_corr_heatmap(method, whiten):

    (X, features, scaler) = Dataset(whiten)
    corr = spearmanr(X).correlation
    # corr_linkage = hierarchy.ward(corr)

    fig = ff.create_dendrogram(
        corr,
        orientation="bottom",
        labels=features,
        linkagefun=hierarchy.ward
    )

    for i in range(len(fig["data"])):
        fig["data"][i]["yaxis"] = "y2"

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(
        corr,
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
    heat_data = corr
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
        Input("checklist-whiten", "value")
    ]
)
def update_components(method, whiten):

    (X, features, scaler) = Dataset(whiten)

    # Form decomposition
    if method == "pca":
        decomposer = decomposition.PCA().fit(X)
    if method == "ica":
        decomposer = decomposition.FastICA(
            random_state=np.random.RandomState(42),
            whiten=False,
            max_iter=1000
        ).fit(X)
    if method == "fa":
        decomposer = decomposition.FactorAnalysis(rotation="varimax").fit(X)
    Y = decomposer.transform(X)
    Y_2d = Y[:, :2]
    U = decomposer.components_
    # V = U
    V = scaler.inverse_transform(U)

    # Fit KDE and sample
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
        rows=3,
        cols=2,
        specs=[
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
            "Two main components (in orignal coordinates)" + (
                ", explained variance: {0:.0%}".format(
                    decomposer.explained_variance_ratio_[:2].sum()
                ) if method == "pca" else ""
            ),
            "Probability density",
            "Samples in 2D",
        ] + (
            ["Samples in original coordinates"] if method in ["pca", "ica"] else
            ["Reverse transformation for factorial analysis not supported :("]
        )
    )
    fig.append_trace(go.Scatter(x=features, y=V[0], name="Component 0"), 1, 1)
    fig.append_trace(go.Scatter(x=features, y=V[1], name="Component 1"), 1, 1)
    fig.append_trace(
        go.Surface(
            x=x,
            y=y,
            z=density,
            showscale=False,
            colorscale="agsunset"
        ),
        2, 1
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
        2, 2
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
        2, 2
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
            2, 2
        )
    if method in ["pca", "ica"]:
        for X_sample in X_samples:
            fig.append_trace(
                go.Scatter(
                    x=features,
                    y=X_sample,
                    showlegend=False
                ),
                3, 1
            )
    if method == "fa":
        fig.append_trace(
            go.Scatter(),
            3, 1
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
