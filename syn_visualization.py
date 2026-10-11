"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Scrollen auf dem Handy nicht zoomt."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import syn_constants as C

_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), height=360, legend=dict(orientation="h", y=-0.25))


def _finish(fig, title, xtitle=None, ytitle=None, **extra):
    fig.update_layout(title=dict(text=title, x=0), xaxis=dict(title=xtitle, fixedrange=True),
                      yaxis=dict(title=ytitle, fixedrange=True), **{**_LAYOUT, **extra})
    return fig


def build_series(one):
    """Servicegrad je Periode: Kontrollen (grau), behandelter Standort (Linie), Synthetische Kontrolle und DiD-Gegenfaktisches,
    dazu das wahre Gegenfaktische (gestrichelt). Prozent."""
    t = np.arange(1, C.N_PERIODS + 1)
    fig = go.Figure()
    yc = one["yc"]
    for k in range(yc.shape[0]):
        fig.add_trace(go.Scatter(x=t, y=100 * yc[k], mode="lines", line=dict(color=C.COLOR_CONTROL, width=1), showlegend=(k == 0),
                                 name="Kontrollstandorte", legendgroup="ctrl", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=100 * one["y0"], mode="lines+markers", name="behandelter Standort",
                             line=dict(color=C.COLOR_TREATED, width=3), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=t, y=100 * one["sc_line"], mode="lines", name="Synthetische Kontrolle",
                             line=dict(color=C.ESTIMATOR_COLORS["sc"], width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=100 * one["did_line"], mode="lines", name="DiD-Gegenfaktisches (Mittel der Kontrollen + Abstand)",
                             line=dict(color=C.ESTIMATOR_COLORS["did"], width=2.5, dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=100 * one["y0_null"], mode="lines", name="wahres Gegenfaktisches (Regel A)",
                             line=dict(color="black", width=2, dash="dash")))
    fig.add_vline(x=C.T0 + 0.5, line=dict(color="gray", width=1.5), annotation_text="neue Regel ab hier", annotation_position="top left")
    fig.update_yaxes(range=[60, 101])
    return _finish(fig, "Servicegrad je Periode (28 Tage)", "Periode", "Servicegrad in %", height=420)


def build_estimates(est, truth):
    """Geschätzte Wirkung je Schätzer in einer Welt (Prozentpunkte) gegen die wahre Wirkung."""
    names = list(C.ESTIMATORS)
    fig = go.Figure(go.Bar(x=[C.ESTIMATOR_SHORT[n] for n in names], y=[100 * est[n] for n in names],
                           marker_color=[C.ESTIMATOR_COLORS[n] for n in names],
                           text=[f"{100 * est[n]:+.2f}".replace(".", ",") for n in names], textposition="outside"))
    fig.add_hline(y=100 * truth, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="wahre Wirkung",
                  annotation_position="top left")
    return _finish(fig, "Geschätzte Wirkung der neuen Regel in dieser Welt", None, "Prozentpunkte Servicegrad")


def build_placebo(taus, tau0, p, j):
    """Placebo-Test für einen Schätzer: |Effekt| der Placebo-Kontrollen (grau, sortiert) und des behandelten Standorts."""
    names = ("did", "sc")
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"{C.ESTIMATOR_SHORT[n]}: p = {p[n]:.2f}".replace(".", ",") for n in names])
    for col, n in enumerate(names, start=1):
        vals = np.sort(np.abs(taus[n]) * 100)[::-1]
        x = [f"K{k + 1}" for k in range(len(vals))]
        fig.add_trace(go.Bar(x=x, y=vals, marker_color=C.COLOR_CONTROL, showlegend=False,
                             hovertemplate="Placebo %{y:.2f} Punkte<extra></extra>"), row=1, col=col)
        fig.add_hline(y=abs(tau0[n]) * 100, line=dict(color=C.COLOR_TREATED, width=3), row=1, col=col,
                      annotation_text="behandelter Standort", annotation_position="top left")
    fig.update_xaxes(showticklabels=False, fixedrange=True, title_text=f"{j} Placebo-Kontrollen, absteigend")
    fig.update_yaxes(fixedrange=True, title_text="|Effekt| in Punkten", col=1)
    fig.update_yaxes(fixedrange=True, col=2)
    fig.update_layout(title=dict(text="Placebo-Test: wie ungewöhnlich ist der Effekt des behandelten Standorts?", x=0), height=340,
                      margin=dict(l=10, r=10, t=80, b=10))
    return fig


def build_bias_by_scenario(scen_labels, rows_by_scen, key, title, ytitle):
    """Gruppierte Balken: je Szenario die vier Schätzer (Verzerrung oder RMSE in Punkten)."""
    fig = go.Figure()
    for est in C.ESTIMATORS:
        fig.add_trace(go.Bar(name=C.ESTIMATOR_SHORT[est], x=scen_labels, y=[rows_by_scen[s][est][key] for s in range(len(scen_labels))],
                             marker_color=C.ESTIMATOR_COLORS[est]))
    fig.update_layout(barmode="group")
    if key == "bias":
        fig.add_hline(y=0, line=dict(color="gray", width=1))
    return _finish(fig, title, None, ytitle)


def build_power(x, series, title, xtitle):
    """Anteil der Welten mit Placebo-p <= Schwelle bei Effekt. series: {Beschriftung: (y-Anteile, Farbe, Strich)}."""
    fig = go.Figure()
    for label, (y, color, dash) in series.items():
        fig.add_trace(go.Scatter(x=x, y=[100 * v for v in y], mode="lines+markers", name=label,
                                 line=dict(color=color, width=3, dash=dash)))
    fig.update_yaxes(range=[0, 100])
    return _finish(fig, title, xtitle, "Anteil der Welten in %")
