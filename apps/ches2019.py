"""Plotly Dash app for displaying analysis results

TODO:
- Separate decomposition figure to multiple callbacks
  - In this way non-linear techniquest, that don't give
    straightforward component vectors, can be supported
- CB: NaN filtering parameters
- Train/test split validation

"""

import argparse
import json
import logging

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy
from scipy.stats import spearmanr

from dimred import analysis, plot, utils
from dimred.datasets.ches2019 import (
    create_dataset,
    download,
    features_bounds,
    load
)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


TIMEOUT = 10


def deserialize(dataset):
    """Deserialize a JSON dataset which has been updated

    """

    if dataset is None:
        raise PreventUpdate

    dataset = json.loads(dataset)
    X = np.asarray(dataset["X"])
    features = dataset["features"]
    scaler = analysis.AffineScaler.from_dict(dataset["scaler"])
    return (X, features, scaler)


def create_app(raw_data):
    """Wrapper for creating the app

    Enables downloading the data to memory just once.

    """

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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
                        html.Div(
                            style={"margin-top": "2em"}
                        ),
                        html.P(
                            "Select dimensionality reduction method"
                        ),
                        dcc.Dropdown(
                            id="dropdown-method",
                            options=[
                                {"label": "Principal component analysis", "value": "pca"},
                                {"label": "Independent component analysis", "value": "ica"},
                                {"label": "Rotated factor analysis", "value": "fa"}
                            ],
                            value="pca",
                            style={"width": "20em"}
                        ),
                        html.P(
                            "Visualize covariance or correlation matrix",
                            style={"margin-top": "1em", "width": "20em"}
                        ),
                        dcc.Dropdown(
                            id="dropdown-corrcov",
                            options=[
                                {"label": "Covariance", "value": "cov"},
                                {"label": "Spearman rank", "value": "corr"},
                            ],
                            value="cov",
                            style={"width": "20em"}
                        ),
                        html.P(
                            "The number of plotted components",
                            style={"margin-top": "1em", "width": "20em"}
                        ),
                        dcc.Input(
                            id="input-components",
                            type="number",
                            placeholder="Number of components visualized",
                            min=1,
                            max=1000,
                            value=2,
                            style={"width": "20em"}
                        )
                    ],
                    style={"float": "left"}
                ),
                html.Div(
                    # TODO: Combine all checklists
                    children=[
                        dcc.Checklist(
                            id="checklist-normalize",
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
                            style={"margin-top": "0.5em", "width": "20em"}
                        ),
                        dcc.Checklist(
                            id="checklist-norint",
                            options=[
                                {"label": "Don't round samples to Int grid", "value": "v"},
                            ],
                            style={"margin-top": "0.5em", "width": "20em"}
                        ),
                    ],
                    style={"float": "left", "margin-left": "2em"}
                )
            ],
            style={"margin-bottom": "20em", "float": "top"}
        ),
        dcc.Loading(
            children=dcc.Graph(id="graph-training-heatmap"),
        ),
        dcc.Loading(
            children=dcc.Graph(id="graph-corr-heatmap"),
        ),
        dcc.Loading(
            children=dcc.Graph(id="graph-components"),
        ),

        # = Data to browser =
        dcc.Store(id="cache")
        # ===================

    ], style={"width": "800px", "margin": "0 auto"})


    @app.callback(
        Output("cache", "data"),
        [
            Input("checklist-normalize", "value"),
            Input("checklist-impute", "value"),
            Input("dropdown-corrcov", "value"),
        ]
    )
    def Dataset(normalize, impute, corrcov):

        (X, features, scaler) = create_dataset(
            data=raw_data,
            normalize=utils.checklist_to_bool(normalize),
            impute=utils.checklist_to_bool(impute),
            corrcov=corrcov
        )

        return json.dumps({
            "features": features,
            "X": X.tolist(),
            "scaler": scaler.to_dict()
        })


    @app.callback(
        Output("graph-training-heatmap", "figure"),
        [
            Input("cache", "data"),
            Input("dropdown-method", "value"),
            Input("checklist-normalize", "value"),
            Input("checklist-impute", "value"),
            Input("dropdown-corrcov", "value"),
        ]
    )
    def update_training_heatmap(dataset, method, normalize, impute, corrcov):

        (X, features, scaler) = deserialize(dataset)

        # Plot dataset
        fig = px.imshow(
            X,
            x=features,
            height=800,
            width=800,
            color_continuous_scale="Agsunset",
            title=(
                "<b>Processed training data, shape: {}</b><br>".format(X.shape) +
                "The features have been grouped with hierarchical clustering (see below image)."
            )
        )

        return fig


    @app.callback(
        Output("graph-corr-heatmap", "figure"),
        [
            Input("cache", "data"),
            Input("dropdown-method", "value"),
            Input("checklist-normalize", "value"),
            Input("checklist-impute", "value"),
            Input("dropdown-corrcov", "value"),
        ]
    )
    def update_corr_heatmap(dataset, method, normalize, impute, corrcov):

        (X, features, scaler) = deserialize(dataset)

        C = np.cov(X.T) if corrcov == "cov" else spearmanr(X).correlation

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
                "Features re-ordered based on hierarchical clustering. The ordering <br>"
                "tries to group together features whose correlation patterns are close <br>"
                "in the Euclidean metric"
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
            Input("cache", "data"),
            Input("dropdown-method", "value"),
            Input("checklist-normalize", "value"),
            Input("checklist-impute", "value"),
            Input("dropdown-corrcov", "value"),
            Input("input-components", "value"),
            Input("checklist-norint", "value"),
        ]
    )
    def update_components(
            dataset,
            method,
            normalize,
            impute,
            corrcov,
            components,
            norint,
    ):

        norint_bool = utils.checklist_to_bool(norint)
        analyze = (
            analysis.analyze_pca if method == "pca" else
            analysis.analyze_ica if method == "ica" else
            analysis.analyze_fa if method == "fa" else
            utils.throw(NotImplementedError("Method not supported"))
        )

        (X, features, scaler) = deserialize(dataset)
        num_samples = 10
        result = analyze(
            X,
            features,
            features_bounds,
            scaler,
            norint=norint_bool,
            components=components,
            num_samples=num_samples
        )
        (U, V, Y_2d, Y_samples, X_samples, statistics) = result["decomposition"]
        (x, y, density, xlim, ylim) = result["density"]
        (min_bounds, max_bounds) = result["reduced_bounds"]

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
                "{} components in normalized coordinate system".format(
                    components
                ) + (
                    ", explained variance: {0:.0%}".format(
                        statistics["explained_variance"]
                    ) if method == "pca" else ""
                ),
                "{0} components in original coordinate system".format(
                    components
                ),
                "Probability density",
                "Samples in 2D",
            ] + (
                [
                    "Principal values" if method == "pca" else "",
                    "Explained variance ratio" if method == "pca" else ""
                ]
            ) + (
                ["Samples in original coordinates\n"] if method == "pca" else
                ["Samples in original coordinates"] if method == "ica" else
                ["Reverse transformation for factor analysis not supported :("]
            )
        )

        #
        # Principal components
        #
        colors = plot.create_colors(components, fmt="hex")
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
        # Principal compoments in scaled coordinates
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
        # FIXME: Whitened PCA somehow messes up the projected bounds
        #

        if (min_bounds is not None) and (max_bounds is not None):

            def add_lines():
                # Need to encapsulate because there is no 'let' in Python :(

                x = np.array([-100, 100])
                widen = lambda x, y: [x - 0.2 * abs(x - y), y + 0.2 * abs(x - y)]
                xlim = widen(min(Y_2d[:, 0]), max(Y_2d[:, 0]))
                ylim = widen(min(Y_2d[:, 1]), max(Y_2d[:, 1]))
                fig.update_xaxes(range=xlim, row=3, col=2)
                fig.update_yaxes(range=ylim, row=3, col=2)

                for (slope, intercept) in min_bounds:
                    y = slope * x + intercept
                    fig.append_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            line=dict(color="blue", width=0.5),
                            showlegend=False
                        ),
                        3, 2
                    )

                for (slope, intercept) in max_bounds:
                    y = slope * x + intercept
                    fig.append_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            line=dict(color="red", width=0.5),
                            showlegend=False
                        ),
                        3, 2
                    )

            ###########
            add_lines()
            ###########


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

        colors = plot.create_colors(num_samples, fmt="hex")
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
            (
                singular_values,
                explained_variance_ratio
            ) = analysis.calculate_pca_statistics(X)
            fig.append_trace(
                go.Scatter(
                    y=singular_values,
                    mode="lines+markers",
                    line={"color": "black", "width": 1.0},
                    showlegend=False
                ),
                4, 1
            )
            fig.append_trace(
                go.Scatter(
                    y=explained_variance_ratio.cumsum(),
                    mode="lines+markers",
                    line={"color": "black", "width": 1.0},
                    showlegend=False
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
        # NOTE: Reverse transform not supported for FactorAnalysis
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

    return app


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Foobar")
    parser.add_argument(
        "--offline",
        help="Run app in offline mode",
        action="store_true"
    )
    args = parser.parse_args()
    offline = args.offline

    try:
        raw_data = load() if offline else download()
    except Exception:
        raw_data = load()
        logging.warning("Something went wrong with downloading data, using cache.")

    app = create_app(raw_data)

    app.run_server(host="0.0.0.0", port="8050", debug=True)
