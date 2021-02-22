import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from flask_caching import Cache
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import decomposition, impute, preprocessing
from sklearn.experimental import enable_iterative_imputer

from dimred import analysis, plot
from dimred.datasets import ches2019


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
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


def Scaler(features):
    return analysis.IntervalScaler([
        v for (k, v) in ches2019.feature_scales.items() if k in features
    ])


@cache.memoize(timeout=TIMEOUT)
def get_training_data():
    x = ches2019.download()  # FIXME FIXME FIXME
    x = ches2019.cleanup(x, nan_floor_row=0.9, nan_floor_col=0.75)
    (X, features) = ches2019.prepare(x)
    imputer = impute.IterativeImputer(max_iter=21).fit(X)
    X = imputer.transform(X)
    return pd.DataFrame(X, columns=features).to_json(orient="split")


def Dataset():
    training_data = pd.read_json(get_training_data(), orient="split")
    X = training_data.values
    features = training_data.columns
    scaler = Scaler(features)
    X = scaler.transform(X)
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
    dcc.Loading(
        children=dcc.Graph(id="graph-heatmap"),
    ),
    dcc.Loading(
        children=dcc.Graph(id="graph-components"),
    )
], style={"width": "800px", "margin": "0 auto"})


@app.callback(
    Output("graph-heatmap", "figure"),
    Input("dropdown-method", "value")
)
def update_heatmap(method):

    (X, features, scaler) = Dataset()

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
    Output("graph-components", "figure"),
    Input("dropdown-method", "value")
)
def update_components(method):

    # Form decomposition
    (X, features, scaler) = Dataset()
    if method == "pca":
        decomposer = decomposition.PCA().fit(X)
    if method == "ica":
        decomposer = decomposition.FastICA(
            random_state=np.random.RandomState(42),
            whiten=True,
            max_iter=1000
        ).fit(X)
    if method == "fa":
        decomposer = decomposition.FactorAnalysis(rotation="varimax").fit(X)
    Y = decomposer.transform(X)
    Y_2d = Y[:, :2]
    U = decomposer.components_
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
            (
                "Two main components, explained variance: {0:.0%}".format(
                    decomposer.explained_variance_ratio_[:2].sum()
                ) if method == "pca" else "Two main components"
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
        height=1200,
        template="plotly_white"
    )

    return fig


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="8050", debug=True)
