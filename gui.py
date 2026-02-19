import numpy as np
from lazy_opt import LazyOpt

hyper_params = {'surrogate':'KNN',
                'epsilon':1,
                'number_of_DOE_samples':100,
                'number_of_iterations':500,
                'number_of_psuedo_candidates':5000
}
options = {'live_plot_draw':False,
           'latent_plot_draw':False,
           'number_of_processes':10,
           'save_csv_filename':'save_csv.csv',
           'save_csv_boolean':False
}
bounds = {f'x{i+1}': (-5, 5) for i in range(15)}

# 2D peaks function (kept for reference)
# bounds = {
#  'x1': (-3, 3),
#  'x2': (-3, 3)
# }
# def function_call(input_row):
#     x1 = input_row[0, 0]
#     x2 = input_row[0, 1]
#     f1 = (3 * (1 - x1) ** 2 * np.exp(-(x1 ** 2) - (x2 + 1) ** 2)
#           - 10 * (x1 / 5 - x1 ** 3 - x2 ** 5) * np.exp(-x1 ** 2 - x2 ** 2)
#           - 1 / 3 * np.exp(-(x1 + 1) ** 2 - x2 ** 2))
#     feasible = f1 < 2
#     return feasible, (f1,)

# 3D peaks function (kept for reference)
# def function_call(input_row):
#     x1, x2, x3 = input_row[0, 0], input_row[0, 1], input_row[0, 2]
#     f1 = (3 * (1 - x1) ** 2 * np.exp(-(x1 ** 2) - (x2 + 1) ** 2)
#           - 10 * (x1 / 5 - x1 ** 3 - x2 ** 5) * np.exp(-x1 ** 2 - x2 ** 2)
#           - 1 / 3 * np.exp(-(x1 + 1) ** 2 - x2 ** 2)
#           + x3 ** 2 * np.exp(-x3 ** 2))
#     feasible = f1 < 2
#     return feasible, (f1,)

def function_call(input_row):
    # 15D Ackley function
    # Global minimum: f=0 at x=(0,...,0)
    # Many local minima — challenging landscape for surrogate-based optimization
    x = input_row[0, :]  # shape (15,)
    n = 15
    a, b, c = 20, 0.2, 2 * np.pi

    sum_sq  = np.sum(x ** 2) / n
    sum_cos = np.sum(np.cos(c * x)) / n

    f1 = -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.e

    # feasible when close enough to the global basin (f < 10 excludes outer ridges)
    feasible = f1 < 10.0

    return feasible, (f1,)

def init_app(lazy):
    # Run this app with `python app.py` and
    # visit http://127.0.0.1:8050/ in your web browser.

    import os
    from dash import Dash, dcc, html
    import plotly.graph_objects as go
    from lazy_plot_langevitour import plot_langevitour

    # Create assets directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)

    # Generate langevitour visualization from the lazy object
    from lazy_plot import latent_pca,latent_tsne,latent_tsne_3d,latent_pca_tsne,latent_knn
    x__pca, xxx__pca = latent_pca(lazy.x_, lazy.xxx_)

    # Get KNN predictions for feasibility
    f_hat_pred_latent = latent_knn(x__pca, xxx__pca, lazy.f)

    # Create feasibility mask (1 = feasible, 0 = infeasible)
    feas = np.array([int(fi[0] < 1e-3) for fi in lazy.f])

    # Get objective values for coloring
    f1 = np.array([obj[0] for obj in lazy.objectives])
    f1_normalized = f1 / np.max(f1)  # Normalize to [0, 1] for colormap

    # Generate PCA+TSNE projection for second plot (like plot_latent)
    print("Generating PCA+TSNE projection...")
    x__tsne = latent_tsne(x__pca)

    # Generate 3D t-SNE on raw design variables for langevitour
    # print("Generating 3D t-SNE for langevitour...")
    # x__tsne_3d = latent_tsne_3d(lazy.x_)

    app = Dash()

    colors = {
        'background': '#FFFFFF',
        'text': '#111111'
    }

    # Create figure and add traces
    fig = go.Figure()

    # Plot xxx points (pseudo-candidates) colored by KNN prediction
    # Green if predicted feasible (f_hat == 0), grey if infeasible
    xxx_feasible = f_hat_pred_latent.ravel() == 0

    print("Generating langevitour visualization...")
    # plot_langevitour_from_lazy_opt(lazy, output_file='assets/lazy_lang_plot.html')
    feas_bool = feas.astype(bool)
    plot_langevitour(x=lazy.x_[feas_bool,:], f=lazy.f[feas_bool], objectives=lazy.objectives, feasible=lazy.f[feas_bool], bounds=None,
                     output_file='assets/lazy_lang_plot.html')


    # Infeasible xxx points (grey)
    fig.add_trace(go.Scattergl(
        x=xxx__pca[~xxx_feasible, 0],
        y=xxx__pca[~xxx_feasible, 1],
        mode='markers',
        name='Predicted Infeasible',
        marker=dict(
            size=6,
            color='grey',
            opacity=0.3
        ),
        showlegend=True
    ))

    # Feasible xxx points (green)
    fig.add_trace(go.Scattergl(
        x=xxx__pca[xxx_feasible, 0],
        y=xxx__pca[xxx_feasible, 1],
        mode='markers',
        name='Predicted Feasible',
        marker=dict(
            size=6,
            color='green',
            opacity=0.5
        ),
        showlegend=True
    ))

    # Plot x points colored by objective value with black edges for feasible
    fig.add_trace(go.Scattergl(
        x=x__pca[:, 0],
        y=x__pca[:, 1],
        mode='markers',
        name='Sampled Points',
        marker=dict(
            size=10,
            color=f1_normalized,
            colorscale='Viridis',
            colorbar=dict(
                title="Objective",
                tickvals=[0, 0.5, 1.0],
                ticktext=[f'{np.min(f1):.2f}', f'{np.mean(f1):.2f}', f'{np.max(f1):.2f}']
            ),
            line=dict(
                color=['black' if fi == 1 else 'rgba(0,0,0,0)' for fi in feas],
                width=2
            ),
            showscale=True
        )
    ))

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        height=700,
        title='Latent Design Space with KNN Boundary (PCA)',
        xaxis_title='Latent x1',
        yaxis_title='Latent x2',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    # Create second figure for PCA+TSNE projection (like plot_latent)
    fig2 = go.Figure()

    # Plot only sampled points colored by objective value with black edges for feasible
    fig2.add_trace(go.Scattergl(
        x=x__tsne[:, 0],
        y=x__tsne[:, 1],
        mode='markers',
        name='Sampled Points',
        marker=dict(
            size=10,
            color=f1_normalized,
            colorscale='Viridis',
            colorbar=dict(
                title="Objective",
                tickvals=[0, 0.5, 1.0],
                ticktext=[f'{np.min(f1):.2f}', f'{np.mean(f1):.2f}', f'{np.max(f1):.2f}']
            ),
            line=dict(
                color=['black' if fi == 1 else 'rgba(0,0,0,0)' for fi in feas],
                width=2
            ),
            showscale=True
        )
    ))

    fig2.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        height=700,
        title='Latent Design Space (PCA+TSNE)',
        xaxis_title='Latent x1',
        yaxis_title='Latent x2',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
        html.Div(style={
            'maxWidth': '1400px',
            'margin': '0 auto',
            'padding': '20px'
        }, children=[

            html.H1(
                children='Hello Dash',
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }
            ),

            html.Div(children='Dash: A web application framework for your data.', style={
                'textAlign': 'center',
                'color': colors['text']
            }),

            html.Div(style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'overflow': 'hidden'
            }, children=[
                html.Iframe(
                    src='/assets/lazy_lang_plot.html',
                    style={
                        "height": "800px",  # Original size
                        "width": "100%",  # Original size
                        "transform": "scale(0.6)",  # Scale to fit
                        "transformOrigin": "center"  # Scale from center
                    }
                )
            ]),

            html.Div(style={'display': 'flex', 'flexDirection': 'row'}, children=[
                html.Div(style={
                    'width': '50%',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'overflow': 'hidden'
                }, children=[
                    dcc.Graph(
                        id='pca-graph',
                        figure=fig,
                        style={"height": "600px", "width": "100%"}
                    )
                ]),

                html.Div(style={
                    'width': '50%',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'overflow': 'hidden'
                }, children=[
                    dcc.Graph(
                        id='pca-tsne-graph',
                        figure=fig2,
                        style={"height": "600px", "width": "100%"}
                    )
                ]),

            ])
        ])
    ])
    return app

if __name__ == "__main__":
    lazy = LazyOpt(solver_function=function_call,
                   bounds=bounds,
                   hyper_params=hyper_params,
                   seed=None,
                   options=options
                   )
    print('=' * 50)
    print('lazy opt finished')
    print('=' * 50)

    app = init_app(lazy)
    app.run(debug=False)