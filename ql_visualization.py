"""Plotly-Abbildungen der Demo "Q-Learning". Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import ql_grid as G

CLIFF_COLOR = "#3a3a3a"
GOAL_COLOR = "#2e7d32"
START_COLOR = "#8c6bb1"
TEXT_LIGHT = "#ffffff"
TEXT_DARK = "#14233B"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def build_grid(grid, V, policy=None):
    """Raster als Heatmap ueber V(s) = max_a Q(s,a), mit Pfeilen der (gierigen) Policy; Klippe dunkel, Ziel gruen, Start violett umrandet - wie in
    `value-iteration-demo`, hier zeigt die Heatmap den aktuellen GELERNTEN Stand statt der exakten Loesung."""
    R, Cc = grid.rows, grid.cols
    z = np.full((R, Cc), np.nan)
    text = [["" for _ in range(Cc)] for _ in range(R)]
    for r in range(R):
        for c in range(Cc):
            s = grid.state_of((r, c))
            kind = grid.cell_kind((r, c))
            if kind == "cliff":
                z[r, c] = np.nan
            else:
                z[r, c] = V[s]
            if kind == "goal":
                text[r][c] = "Ziel"
            elif kind == "cliff":
                text[r][c] = ""
            elif policy is not None:
                text[r][c] = G.ACTION_ARROWS[policy[s]]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=z, colorscale="RdYlGn", zmid=0, showscale=True, text=[[de(v, 1) if not np.isnan(v) else "" for v in row] for row in z],
                              hovertemplate="Zeile %{y}, Spalte %{x}: Q=%{z:.2f}<extra></extra>", colorbar=dict(title="max Q(s,·)", thickness=14)))
    cliff_x = [c for r in range(R) for c in range(Cc) if grid.cell_kind((r, c)) == "cliff"]
    cliff_y = [r for r in range(R) for c in range(Cc) if grid.cell_kind((r, c)) == "cliff"]
    if cliff_x:
        fig.add_trace(go.Scatter(x=cliff_x, y=cliff_y, mode="markers", marker=dict(symbol="square", size=34, color=CLIFF_COLOR), showlegend=False, hovertemplate="Klippe<extra></extra>"))
    for r in range(R):
        for c in range(Cc):
            kind = grid.cell_kind((r, c))
            if text[r][c]:
                color = TEXT_LIGHT if kind == "goal" else TEXT_DARK
                fig.add_annotation(x=c, y=r, text=text[r][c], showarrow=False, font=dict(size=18, color=color))
    sr, sc = grid.start
    fig.add_shape(type="rect", x0=sc - 0.45, x1=sc + 0.45, y0=sr - 0.45, y1=sr + 0.45, line=dict(color=START_COLOR, width=3))
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return _base(fig, 90 * grid.rows + 60)


def build_learning_curve(returns, V_star_start, window=50):
    """Ertrag je Episode (gleitender Durchschnitt), mit V*(Start) als Referenzlinie."""
    y = np.asarray(returns, dtype=float)
    x = np.arange(1, len(y) + 1)
    if len(y) >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(y, kernel, mode="valid")
        x_smooth = x[window - 1:]
    else:
        smooth, x_smooth = y, x
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color="#c7d4e6", size=3), name="Ertrag je Episode", hovertemplate="Episode %{x}: %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x_smooth, y=smooth, mode="lines", line=dict(color="#1f77b4", width=2), name=f"Gleitender Durchschnitt ({window})"))
    fig.add_hline(y=V_star_start, line=dict(color="#2e7d32", width=2, dash="dash"), annotation_text="V*(Start)", annotation_position="bottom right")
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Ertrag")
    return _base(fig, 300).update_layout(legend=dict(orientation="h", y=-0.3))


def build_episodes_gap(exp, vi_backups):
    x = [r["episodes"] for r in exp["rows"]]
    mean = [r["mean"] for r in exp["rows"]]
    se = [r["se"] for r in exp["rows"]]
    steps = [r["env_steps_mean"] for r in exp["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=mean, error_y=dict(type="data", array=se), mode="lines+markers", line=dict(color="#d62728", width=2), marker=dict(size=7), name="V*(Start) − V_gelernt(Start)"))
    fig.add_trace(go.Scatter(x=x, y=steps, mode="lines+markers", line=dict(color="#1f77b4", width=2, dash="dot"), marker=dict(size=6), name="Umgebungsschritte (Mittel)", yaxis="y2"))
    fig.add_hline(y=vi_backups, line=dict(color="#2e7d32", width=2, dash="dash"), yref="y2", annotation_text=f"Value Iteration: {vi_backups} Bellman-Backups", annotation_position="top left")
    fig.update_layout(yaxis=dict(title="Wert-Abstand", rangemode="tozero"), yaxis2=dict(title="Umgebungsschritte", overlaying="y", side="right", type="log"))
    fig.update_xaxes(title_text="Trainingsepisoden")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.3))


def build_decay(exp):
    x = [de(r["decay"], 3) for r in exp["rows"]]
    mean = [r["mean"] for r in exp["rows"]]
    se = [r["se"] for r in exp["rows"]]
    frac = [100 * r["frac_near_optimal"] for r in exp["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=mean, error_y=dict(type="data", array=se), name="Wert-Abstand (Mittel)", marker=dict(color="#d62728"), yaxis="y1"))
    fig.add_trace(go.Scatter(x=x, y=frac, name="Anteil nahe optimal (%)", mode="lines+markers", line=dict(color="#1f77b4", width=2), marker=dict(size=7), yaxis="y2"))
    fig.update_layout(yaxis=dict(title="Wert-Abstand", rangemode="tozero"), yaxis2=dict(title="Anteil nahe optimal (%)", overlaying="y", side="right", range=[0, 100], showgrid=False))
    fig.update_xaxes(title_text="Epsilon-Zerfall (0 = konstant explorativ)", type="category")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))


def build_alpha(exp):
    x = [de(r["alpha"], 2) for r in exp["rows"]]
    mean = [r["mean"] for r in exp["rows"]]
    se = [r["se"] for r in exp["rows"]]
    frac = [100 * r["frac_near_optimal"] for r in exp["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=mean, error_y=dict(type="data", array=se), name="Wert-Abstand (Mittel)", marker=dict(color="#d62728"), yaxis="y1"))
    fig.add_trace(go.Scatter(x=x, y=frac, name="Anteil nahe optimal (%)", mode="lines+markers", line=dict(color="#1f77b4", width=2), marker=dict(size=7), yaxis="y2"))
    fig.update_layout(yaxis=dict(title="Wert-Abstand", rangemode="tozero"), yaxis2=dict(title="Anteil nahe optimal (%)", overlaying="y", side="right", range=[0, 100], showgrid=False))
    fig.update_xaxes(title_text="Lernrate α", type="category")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))
