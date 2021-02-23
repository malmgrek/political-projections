"""Plotly Dash app for displaying analysis results

TODO:
- CB: NaN filtering parameters
- Train/test split validation

"""

import random

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from flask_caching import Cache
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn import decomposition

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
    scales = ches2019.features_bounds
    return analysis.IntervalScaler([scales[f] for f in features])


def reorder_features(X, features, corrcov):
    #
    # TODO/FIXME/HACK: This is an ugly workaround
    # for ordering the features as in the dendrogram
    #
    C = np.cov(X.T) if corrcov == "cov" else spearmanr(X).correlation
    fig = ff.create_dendrogram(
        C,
        orientation="bottom",
        labels=features,
        linkagefun=hierarchy.ward
    )
    new_features = list(fig["layout"]["xaxis"]["ticktext"])
    x = pd.DataFrame(X, columns=features)
    x = x[new_features]
    return (x.values, new_features)


@cache.memoize(timeout=TIMEOUT)
def get_training_data():
    # x = ches2019.download()  # FIXME
    x = ches2019.load()
    x = ches2019.cleanup(x, nan_floor_row=0.9, nan_floor_col=0.75)
    (X, features) = ches2019.prepare(x)
    return pd.DataFrame(X, columns=features).to_json(orient="split")


def Dataset(meanvar=None, impute=None, corrcov=None):

    meanvar_bool = checklist_to_bool(meanvar)
    impute_bool = checklist_to_bool(impute)

    training_data = pd.read_json(get_training_data(), orient="split")

    # Optionally impute
    X_train = (
        analysis.impute_missing(training_data.values, max_iter=21) if impute_bool
        else training_data.dropna().values
    )
    features = list(training_data.columns)

    #
    # NOTE: We first reorder features with hierarchical clustering analysis
    # and then define the scaler. The former should be more or less independent
    # of scaling. It seems to make sense to scale before and after reordering.
    #

    def create_scaler(_X, _features):
        return (
            analysis.StandardScaler().fit(_X) if meanvar_bool else
            IntervalScaler(_features)
        )

    find_permutation = lambda xs, ys: [xs.index(y) for y in ys]

    # Scale once before re-ordering
    X = create_scaler(X_train, features).transform(X_train)
    # Re-order features
    (X_reord, features_reord) = reorder_features(X, features, corrcov)
    X_train_reord = X_train[:, find_permutation(features, features_reord)]
    # Create new scaler for further use using the re-ordered features set
    scaler = create_scaler(X_train_reord, features_reord)
    # Scale training data
    X = scaler.transform(X_train_reord)

    return (X, features_reord, scaler)


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
                    )
                ],
                style={"float": "left"}
            ),
            html.Div(
                children=[
                    dcc.Checklist(
                        id="checklist-meanvar",
                        options=[
                            {"label": "Map to zero mean and unit variance", "value": "v"},
                        ],
                        style={"margin-top": "2em", "width": "20em"}
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
                    dcc.Checklist(
                        id="checklist-norint",
                        options=[
                            {"label": "Don't round samples to Int grid", "value": "v"},
                        ],
                        style={"margin-top": "1em", "width": "20em"}
                    ),
                ],
                style={"float": "left", "margin-left": "2em"}
            )
        ],
        style={"margin-bottom": "15em", "float": "top"}
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
        linkagefun=hierarchy.ward,
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
            colorscale="Agsunset",
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
        "template": "plotly_white",
        "title": (
            "<b>Covariance (or correlation plot)</b> <br>"
            "Features re-ordered based on hierarchical clustering"
        )
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
        Input("checklist-norint", "value"),
    ]
)
def update_components(
        method,
        meanvar,
        impute,
        whiten,
        corrcov,
        components,
        norint
):

    whiten_bool = checklist_to_bool(whiten)
    norint_bool = checklist_to_bool(norint)

    (X, features, scaler) = Dataset(meanvar, impute, corrcov)

    bounds = np.array([ches2019.features_bounds[f] for f in features])

    #
    # Form decomposition
    #
    if method == "pca":
        decomposer = decomposition.PCA(whiten=whiten_bool).fit(X)
    if method == "ica":
        decomposer = decomposition.FastICA(
            random_state=np.random.RandomState(42),
            whiten=whiten_bool,
            max_iter=500,
            tol=1e-3
        ).fit(X)
    if method == "fa":
        decomposer = decomposition.FactorAnalysis(rotation="varimax").fit(X)
    Y = decomposer.transform(X)
    Y_2d = Y[:, :2]
    U = decomposer.components_
    V = scaler.inverse_transform(U)

    #
    # Fit KDE and sample
    # TODO: Cap samples after inverse transforming
    #
    kde = analysis.fit_kde(Y_2d)
    num_samples = 10
    Y_samples = kde.sample(num_samples)
    (x, y, density, xlim, ylim) = analysis.score_density_grid(
        kde=kde, Y=Y_2d, num=100
    )

    #
    # Inverse transform not supported for FA :(
    #
    Y_samples_full = np.hstack((
        Y_samples,
        np.zeros((num_samples, len(features) - 2))
    ))
    if method in ["pca", "ica"]:
        X_samples = scaler.inverse_transform(
            decomposer.inverse_transform(Y_samples_full)
        )
        X_samples = (
            X_samples if norint_bool
            else np.clip(np.rint(X_samples), *bounds.T)
        )

    # TODO: Add one more row with principal components
    fig = make_subplots(
        rows=5,
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
                {"rowspan": 1, "colspan": 1},
                {"rowspan": 1, "colspan": 1},
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
            [
                "Principal values" if method == "pca" else "",
                "Explained variance ratio" if method == "pca" else ""
            ]
        ) + (
            [
                "Samples in original coordinates\n"
                " (up to a linear combo of remaining components)"
            ] if method == "pca" else
            ["Samples in original coordinates"] if method == "ica" else
            ["Reverse transformation for factorial analysis not supported :("]
        )
    )

    #
    # Principal components
    #
    colors = random.choices(px.colors.qualitative.Plotly, k=components)
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

    #
    # Principal compoments in scalet coordinates
    #
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

    #
    # 3d surface plot of estimated density
    #
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


    #
    # Projected limit bounds
    #
    if method == "pca":

        def add_projected_lines():

            rotate = lambda x: np.dot(U, x)
            translate = lambda x, n: x - np.dot(X.mean(axis=0), n) * n
            n_dims = len(features)
            bounds_scaled = scaler.transform(bounds.T).T

            for (i, (a, b)) in enumerate(bounds_scaled):

                for (c, color) in zip((a, b), ("blue", "red")):

                    n_vec = (np.arange(n_dims) == i) * c
                    a_vec = n_vec

                    #
                    # PCA transformation first shifs to zero mean and then rotates.
                    # For plane's vector geometry it means that the NORMAL VECTOR is
                    # just rotated and the OFFSET VECTOR is
                    #
                    # (1) translated in the plane normal direction
                    # (2) rotated by the rotation
                    #
                    a_vec = rotate(translate(a_vec, n_vec))
                    n_vec = rotate(n_vec)
                    (slope, intercept) = analysis.intersect_plane_xy(n_vec, a_vec)
                    x = np.array([-100, 100])
                    y = slope * x + intercept

                    fig.append_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            line=dict(color=color, width=0.5),
                            showlegend=False
                        ),
                        3, 2
                    )
                    widen = lambda x, y: [x - 0.2 * abs(x - y), y + 0.2 * abs(x - y)]
                    xlim = widen(min(Y_2d[:, 0]), max(Y_2d[:, 0]))
                    ylim = widen(min(Y_2d[:, 1]), max(Y_2d[:, 1]))
                    fig.update_xaxes(range=xlim, row=3, col=2)
                    fig.update_yaxes(range=ylim, row=3, col=2)

            return

        # ===================
        add_projected_lines()
        # ===================

    #
    # 2d representation of density and projected points
    #
    fig.append_trace(
        go.Scatter(
            x=Y_2d[:, 0],
            y=Y_2d[:, 1],
            mode="markers",
            marker={
                "color": "black",
                "size": 12,
                "opacity": 0.3
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

    colors = random.choices(px.colors.qualitative.Plotly, k=num_samples)
    for (i, (Y_sample, color)) in enumerate(zip(Y_samples, colors)):
        fig.append_trace(
            go.Scatter(
                x=[Y_sample[0]],
                y=[Y_sample[1]],
                mode="markers",
                marker={"size": 18, "color": color},
                name="Sample {}".format(i)
            ),
            3, 2
        )

    #
    # Principal values
    #
    if method == "pca":
        fig.append_trace(
            go.Scatter(
                y=decomposer.singular_values_,
                mode="lines+markers",
                line={"color": "black", "width": 1.0},
                # marker={"style": "o"}
            ),
            4, 1
        )
        fig.append_trace(
            go.Scatter(
                y=decomposer.explained_variance_ratio_.cumsum(),
                mode="lines+markers",
                line={"color": "black", "width": 1.0},
                # marker={"style": "o"}
            ),
            4, 2
        )




    #
    # Reverse transformed samples
    #
    if method in ["pca", "ica"]:
        for (i, (X_sample, color)) in enumerate(zip(X_samples, colors)):
            fig.append_trace(
                go.Scatter(
                    x=features,
                    y=X_sample,
                    showlegend=False,
                    line={"color": color},
                    name="Sample {}".format(i)
                ),
                5, 1
            )
    # NOTE: Reverse transform not supported for FactorialAnalysis
    if method == "fa":
        fig.append_trace(
            go.Scatter(),
            5, 1
        )

    fig.update_layout(
        autosize=False,
        width=1000,
        height=2000,
        template="plotly_white"
    )

    return fig


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="8050", debug=True)
