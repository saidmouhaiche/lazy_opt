import time
import numpy as np
from lazy_opt import lazy_opt

# # # 2D peaks function (kept for reference)
# bounds = {
#     'x1': (-3, 3),
#     'x2': (-3, 3)
# }
# def function_call(input_row):
#     x1 = input_row[0, 0]
#     x2 = input_row[0, 1]
#     f1 = (3 * (1 - x1) ** 2 * np.exp(-(x1 ** 2) - (x2 + 1) ** 2)
#           - 10 * (x1 / 5 - x1 ** 3 - x2 ** 5) * np.exp(-x1 ** 2 - x2 ** 2)
#           - 1 / 3 * np.exp(-(x1 + 1) ** 2 - x2 ** 2))
#     feasible = f1 < 2
#     return feasible, (f1,)

# # 3D peaks function (kept for reference)
# bounds = {
#     'x1': (-3, 3),
#     'x2': (-3, 3),
#     'x3': (-3, 3)
# }
# def function_call(input_row):
#     x1, x2, x3 = input_row[0, 0], input_row[0, 1], input_row[0, 2]
#     f1 = (3 * (1 - x1) ** 2 * np.exp(-(x1 ** 2) - (x2 + 1) ** 2)
#           - 10 * (x1 / 5 - x1 ** 3 - x2 ** 5) * np.exp(-x1 ** 2 - x2 ** 2)
#           - 1 / 3 * np.exp(-(x1 + 1) ** 2 - x2 ** 2)
#           + x3 ** 2 * np.exp(-x3 ** 2))
#     feasible = f1 < 2
#     return feasible, (f1,)

# bounds = {f'x{i+1}': (-5, 5) for i in range(15)}
# def function_call(input_row):
#     # 15D Ackley function
#     # Global minimum: f=0 at x=(0,...,0)
#     # Many local minima — challenging landscape for surrogate-based optimization
#     x = input_row[0, :]  # shape (15,)
#     n = 15
#     a, b, c = 20, 0.2, 2 * np.pi
#
#     sum_sq  = np.sum(x ** 2) / n
#     sum_cos = np.sum(np.cos(c * x)) / n
#
#     f1 = -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.e
#
#     # feasible when close enough to the global basin (f < 10 excludes outer ridges)
#     feasible = f1 < 10.0
#
#     return feasible, (f1,)

bounds = {
    'x1': (0, 1),
    'x2': (0, 1),
    'x3': (0, 1),
    'x4': (0, 1),
}

def function_call(input_row):
    x = input_row[0, :]

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10.0,  3.0, 17.0,  3.5],
        [ 0.05, 10.0, 17.0,  0.1],
        [ 3.0,  3.5,  1.7, 10.0],
        [17.0,  8.0,  0.05, 10.0],
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569,  124],
        [2329, 4135, 8307, 3736],
        [2348, 1451, 3522, 2883],
        [4047, 8828, 8732, 5743],
    ])

    f1 = -np.sum(alpha * np.exp(-np.sum(A * (x - P) ** 2, axis=1)))

    # Feasible when in the lower half of the objective landscape.
    # The global minimum is ≈ -3.13; values below -1.0 form a well-posed
    # feasible region that includes the optimum but excludes most of the domain.
    feasible = f1 < -1.0

    return feasible, (f1,)



# ══════════════════════════════════════════════════════════════════════════════
# USER-CUSTOMIZABLE GEOMETRY FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
# Replace the body of draw_geometry() with your own rendering logic.
#
# Called once per selected design.  Add traces to `fig` at the given subplot
# cell — do NOT create a new figure.  Use only go.Scatter / go.Scatter3d etc.
#
# Parameters
# ----------
# x             : np.ndarray, shape (dims,), design variables in their raw units
# fig           : plotly.graph_objects.Figure  (a make_subplots figure)
# subplot_row   : int, 1-based row index of the target subplot cell
# subplot_col   : int, 1-based column index of the target subplot cell
# ══════════════════════════════════════════════════════════════════════════════
def draw_geometry(x, fig, subplot_row, subplot_col):
    import numpy as np
    import plotly.graph_objects as go

    x = np.asarray(x, dtype=float)

    def get(i, default=0.5):
        return float(x[i]) if len(x) > i else default

    # ── Map design variables → rocket shape parameters ────────────────────────
    nose_len  = 0.5 + get(0) * 1.5     # nose-cone length    (0.5 – 2.0)
    body_len  = 3.0 + get(1) * 4.0     # body tube length    (3.0 – 7.0)
    fin_span  = 0.4 + get(2) * 0.8     # fin half-span (×R)  (0.4 – 1.2)
    fin_sweep = 20  + get(3) * 50      # fin sweep angle °   (20  – 70)

    R = 0.5   # body radius — fixed display scale

    # ── Body (rectangle = cylinder side-view) ─────────────────────────────────
    bx = [-R,  R,  R, -R, -R]
    by = [ 0,  0,  body_len, body_len, 0]

    # ── Nose cone (triangle) ──────────────────────────────────────────────────
    nx = [-R, 0,  R, -R]
    ny = [body_len, body_len + nose_len, body_len, body_len]

    # ── Fins (trapezoidal, swept leading edge) ────────────────────────────────
    fin_h    = R * 1.2
    sweep_dx = fin_h / np.tan(np.radians(max(fin_sweep, 1)))
    span     = R * fin_span

    lf_x = [-R, -R - span, -R - span + sweep_dx, -R, -R]
    lf_y = [ 0,  0,         fin_h,                fin_h, 0]

    rf_x = [ R,  R + span,  R + span - sweep_dx,  R,  R]
    rf_y = [ 0,  0,         fin_h,                 fin_h, 0]

    kw     = dict(row=subplot_row, col=subplot_col)
    shared = dict(mode='lines', fill='toself', showlegend=False)

    fig.add_trace(go.Scatter(x=bx,    y=by,    name='body',
                              line=dict(color='steelblue', width=1.5),
                              fillcolor='rgba(70,130,180,0.35)', **shared), **kw)
    fig.add_trace(go.Scatter(x=nx,    y=ny,    name='nose',
                              line=dict(color='steelblue', width=1.5),
                              fillcolor='rgba(70,130,180,0.55)', **shared), **kw)
    fig.add_trace(go.Scatter(x=lf_x,  y=lf_y,  name='fin-L',
                              line=dict(color='firebrick', width=1.5),
                              fillcolor='rgba(178,34,34,0.45)', **shared), **kw)
    fig.add_trace(go.Scatter(x=rf_x,  y=rf_y,  name='fin-R',
                              line=dict(color='firebrick', width=1.5),
                              fillcolor='rgba(178,34,34,0.45)', **shared), **kw)
# ══════════════════════════════════════════════════════════════════════════════


def init_app(lazy):
    # Run this app with `python app.py` and
    # visit http://127.0.0.1:8050/ in your web browser.

    import os
    from dash import Dash, dcc, html, dash_table, Input, Output
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from lazy_plot import plot_langevitour, plot_langevitour_with_hex_model

    # Create assets directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)

    # Generate langevitour visualization from the lazy object
    from lazy_plot import latent_pca,latent_tsne,latent_tsne_3d,latent_pca_tsne,latent_knn,latent_classifier
    x__pca, xxx__pca = latent_pca(lazy.x_, lazy.xxx_)

    # Get KNN predictions for feasibility
    # f_hat_pred_latent = latent_knn(x__pca, xxx__pca, lazy.f)
    f_hat_pred_latent = latent_classifier(lazy, x__pca, xxx__pca, lazy.f)

    # Create feasibility mask (1 = feasible, 0 = infeasible)
    feas = np.array([int(fi[0] < 1e-3) for fi in lazy.f])

    # Get objective values for coloring
    f1 = np.array([obj[0] for obj in lazy.objectives])
    f1_normalized = (f1 - np.min(f1)) / (np.max(f1) - np.min(f1))  # Normalize to [0, 1]

    iteration_numbers_arr = np.array(lazy.iteration_numbers)
    min_iter = int(np.min(iteration_numbers_arr))
    max_iter = int(np.max(iteration_numbers_arr))

    # Build table column definitions from lazy dimensionality
    dims = lazy.hyper_params[6]
    var_names = [f'x{i+1}' for i in range(dims)]
    table_columns = (
        [{'name': 'index', 'id': 'index'}]
        + [{'name': n, 'id': n} for n in var_names]
        + [{'name': 'f1', 'id': 'f1'}, {'name': 'feasible', 'id': 'feasible'}, {'name': 'iteration', 'id': 'iteration'}]
    )

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

    print("Generating langevitour visualization with hex model...")
    feas_mask = feas.astype(bool)
    plot_langevitour_with_hex_model(
        high_d_data=lazy.x_[feas_mask],
        layout_2d=x__tsne[feas_mask],
        output_file='assets/lazy_lang_plot.html',
    )


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
            cmin=0, cmax=1,
            colorbar=dict(
                title="Objective",
                tickvals=[0, 0.5, 1.0],
                ticktext=[f'{np.min(f1):.2f}', f'{(np.min(f1)+np.max(f1))/2:.2f}', f'{np.max(f1):.2f}']
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

    # Helper: build PCA+TSNE figure filtered to iterations 0..max_it
    def make_tsne_fig(max_it):
        mask = iteration_numbers_arr <= max_it
        f = go.Figure()
        f.add_trace(go.Scattergl(
            x=x__tsne[mask, 0],
            y=x__tsne[mask, 1],
            mode='markers',
            name='Sampled Points',
            marker=dict(
                size=10,
                color=f1_normalized[mask],
                colorscale='Viridis',
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title="Objective",
                    tickvals=[0, 0.5, 1.0],
                    ticktext=[f'{np.min(f1):.2f}', f'{np.mean(f1):.2f}', f'{np.max(f1):.2f}']
                ),
                line=dict(
                    color=['black' if fi == 1 else 'rgba(0,0,0,0)' for fi in feas[mask]],
                    width=2
                ),
                showscale=True
            )
        ))
        f.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text'],
            height=700,
            title=f'Latent Design Space (PCA+TSNE) — Iterations {min_iter}–{max_it}',
            xaxis_title='Latent x1',
            yaxis_title='Latent x2',
            dragmode='select',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        return f

    fig2 = make_tsne_fig(max_iter)

    # Pre-compute fixed jitter positions so points don't shift on selection change
    rng = np.random.default_rng(42)
    violin_jitter = [rng.uniform(-0.2, 0.2, lazy.x.shape[0]) for _ in range(dims)]

    def make_violin_fig(selected_indices):
        fig = make_subplots(
            rows=1, cols=dims,
            subplot_titles=var_names,
            shared_yaxes=False,
        )
        for col_idx, var_name in enumerate(var_names):
            all_vals = lazy.x[:, col_idx]
            jitter = violin_jitter[col_idx]
            is_first = col_idx == 0

            # Violin shape (full distribution)
            fig.add_trace(go.Violin(
                y=all_vals,
                x0=0,
                name=var_name,
                showlegend=False,
                points=False,
                line_color='#4C72B0',
                fillcolor='rgba(76,114,176,0.25)',
                meanline_visible=True,
                width=0.8,
            ), row=1, col=col_idx + 1)

            # All points — low opacity
            fig.add_trace(go.Scatter(
                x=jitter,
                y=all_vals,
                mode='markers',
                name='All Points',
                showlegend=is_first,
                legendgroup='all',
                marker=dict(size=5, color='#4C72B0', opacity=0.2),
            ), row=1, col=col_idx + 1)

            # Selected points — high opacity
            if selected_indices:
                sel_vals = lazy.x[selected_indices, col_idx]
                sel_jitter = jitter[selected_indices]
                fig.add_trace(go.Scatter(
                    x=sel_jitter,
                    y=sel_vals,
                    mode='markers',
                    name='Selected',
                    showlegend=is_first,
                    legendgroup='selected',
                    marker=dict(
                        size=9,
                        color='crimson',
                        opacity=0.95,
                        line=dict(color='black', width=1)
                    ),
                ), row=1, col=col_idx + 1)

            fig.update_xaxes(
                showticklabels=False, showgrid=False, zeroline=False,
                row=1, col=col_idx + 1
            )
            fig.update_yaxes(
                showgrid=True, gridcolor='#e0e0e0',
                row=1, col=col_idx + 1
            )

        fig.update_layout(
            height=320,
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['background'],
            font_color=colors['text'],
            title='Variable Distributions (select points on PCA+TSNE to highlight)',
            margin=dict(t=60, b=20, l=40, r=20),
        )
        return fig

    def make_geometry_fig(rows_data):
        """Build a figure showing up to 5 design geometries from table row dicts."""
        n = min(5, len(rows_data)) if rows_data else 0
        if n == 0:
            empty = go.Figure()
            empty.update_layout(
                height=300,
                paper_bgcolor=colors['background'],
                plot_bgcolor=colors['background'],
                font_color=colors['text'],
                title='Geometry Viewer (select points on the PCA+TSNE plot)',
                annotations=[dict(
                    text='No points selected', x=0.5, y=0.5,
                    xref='paper', yref='paper',
                    showarrow=False, font=dict(size=16, color='#aaa'),
                )],
            )
            return empty

        titles = [f"idx {rows_data[i]['index']}" for i in range(n)]
        gfig = make_subplots(rows=1, cols=n, subplot_titles=titles)

        for i, row in enumerate(rows_data[:5]):
            x_row = np.array([row[name] for name in var_names], dtype=float)
            draw_geometry(x_row, gfig, subplot_row=1, subplot_col=i + 1)

            # Equal-aspect ratio so the rocket isn't squashed
            xref = 'x' if i == 0 else f'x{i + 1}'
            gfig.update_yaxes(
                scaleanchor=xref, scaleratio=1,
                showgrid=False, zeroline=False, showticklabels=False,
                row=1, col=i + 1,
            )
            gfig.update_xaxes(
                showgrid=False, zeroline=False, showticklabels=False,
                row=1, col=i + 1,
            )

        gfig.update_layout(
            height=380,
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['background'],
            font_color=colors['text'],
            title='Geometry Viewer — First 5 Selected Points',
            showlegend=False,
            margin=dict(t=60, b=10, l=10, r=10),
        )
        return gfig

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

            # Row 1: PCA plot + Langevitour side by side
            html.Div(style={'display': 'flex', 'flexDirection': 'row'}, children=[
                html.Div(style={
                    'width': '50%',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
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
                    html.Iframe(
                        src='/assets/lazy_lang_plot.html',
                        style={
                            "height": "800px",
                            "width": "100%",
                            "transform": "scale(0.6)",
                            "transformOrigin": "center"
                        }
                    )
                ]),
            ]),

            # Row 2: PCA+TSNE plot by itself (brush selection feeds table)
            html.Div(style={'marginTop': '20px', 'padding': '0 40px'}, children=[
                html.Label(
                    'Show iterations up to:',
                    style={'fontWeight': 'bold', 'color': colors['text']}
                ),
                dcc.Slider(
                    id='iteration-slider',
                    min=min_iter,
                    max=max_iter,
                    step=1,
                    value=max_iter,
                    marks={i: str(i) for i in range(
                        min_iter,
                        max_iter + 1,
                        max(1, max_iter // 10)
                    )} | {max_iter: str(max_iter)},
                    tooltip={"placement": "bottom", "always_visible": True},
                )
            ]),
            html.Div(style={'display': 'flex', 'justifyContent': 'center'}, children=[
                dcc.Graph(
                    id='pca-tsne-graph',
                    figure=fig2,
                    style={"height": "700px", "width": "100%"}
                )
            ]),

            # Variable distribution violin plots
            html.Div(style={'marginTop': '10px'}, children=[
                dcc.Graph(
                    id='violin-graph',
                    figure=make_violin_fig([]),
                    style={"height": "320px", "width": "100%"}
                )
            ]),

            # Geometry viewer — first 5 designs from the sorted selection table
            html.Div(style={'marginTop': '10px'}, children=[
                dcc.Graph(
                    id='geometry-graph',
                    figure=make_geometry_fig([]),
                    style={"height": "380px", "width": "100%"},
                )
            ]),

            html.Div(style={'marginTop': '20px'}, children=[
                html.H3('Selected Points', style={'textAlign': 'center', 'color': colors['text']}),
                dash_table.DataTable(
                    id='data-table',
                    columns=table_columns,
                    data=[],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px', 'fontFamily': 'monospace'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
                    page_size=20,
                    sort_action='native',
                )
            ]),

        ])
    ])

    @app.callback(
        Output('pca-tsne-graph', 'figure'),
        Input('iteration-slider', 'value'),
    )
    def update_tsne_plot(max_it):
        return make_tsne_fig(max_it)

    @app.callback(
        Output('violin-graph', 'figure'),
        Input('pca-tsne-graph', 'selectedData'),
    )
    def update_violin(selected):
        if not selected:
            return make_violin_fig([])
        indices = [
            p['pointIndex'] for p in selected['points']
            if p.get('curveNumber') == 0
        ]
        return make_violin_fig(indices)

    @app.callback(
        Output('geometry-graph', 'figure'),
        Input('data-table', 'derived_virtual_data'),
    )
    def update_geometry(rows_data):
        return make_geometry_fig(rows_data)

    @app.callback(
        Output('data-table', 'data'),
        Input('pca-tsne-graph', 'selectedData'),
    )
    def update_table(selected):
        if not selected:
            return []
        # curveNumber 0 is the only trace in fig2 ('Sampled Points')
        indices = [
            p['pointIndex'] for p in selected['points']
            if p.get('curveNumber') == 0
        ]
        rows = []
        for i in sorted(indices):
            row = {'index': i}
            for j, name in enumerate(var_names):
                row[name] = round(float(lazy.x[i, j]), 4)
            row['f1'] = round(float(lazy.objectives[i][0]), 4)
            # strangely, feasible==0 means that it is feasible (minimizing problem)
            row['feasible'] = 'No' if lazy.feasible[i] else 'Yes'
            row['iteration'] = int(lazy.iteration_numbers[i])
            rows.append(row)
        return rows

    return app

if __name__ == "__main__":
    hyper_params = {'surrogate': 'hnswlib',
                    'epsilon': 2,
                    'number_of_DOE_samples': 10,
                    'number_of_iterations': 100,
                    'number_of_psuedo_candidates': 1000
                    }
    options = {'live_plot_draw': False,
               'latent_plot_draw': False,
               'number_of_processes': 1,
               'save_csv_filename': 'save_csv.csv',
               'save_csv_boolean': False,
               'verbose': False
               }
    t0 = time.time()
    lazy = lazy_opt(solver_function=function_call,
                    bounds=bounds,
                    hyper_params=hyper_params,
                    seed=None,
                    options=options
                    )
    elapsed = time.time() - t0
    print('=' * 50)
    print('lazy opt finished')
    print(f'time elapsed: {elapsed:.2f}s')
    print('=' * 50)

    app = init_app(lazy)
    app.run(debug=False)