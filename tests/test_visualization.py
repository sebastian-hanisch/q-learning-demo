"""Regressionstest fuer einen echten, im Browser gefundenen Bug: Plotly.js interpretiert deutsch formatierte Dezimal-Strings ("0,10") als
US-tausendergruppierte Ganzzahlen (10), wenn die x-Achse numerisch statt kategorial ist - die Achse zeigte dann "0/5/10/20/30" statt "0,00/0,05/0,10/0,20/0,30".
Behoben mit explizitem type="category"; dieser Test haelt die Achsen-Beschriftung fest, damit der Fehler nicht unbemerkt zurueckkommt."""

import ql_evaluation as E
from ql_visualization import build_alpha, build_decay


def test_decay_chart_x_axis_is_categorical_not_silently_numeric():
    exp = E.decay_experiment(levels=(0.0, 0.01), seeds=range(3))
    fig = build_decay(exp)
    assert fig.layout.xaxis.type == "category"
    assert list(fig.data[0].x) == ["0,000", "0,010"]


def test_alpha_chart_x_axis_is_categorical_not_silently_numeric():
    exp = E.alpha_experiment(levels=(0.10, 0.30), seeds=range(3))
    fig = build_alpha(exp)
    assert fig.layout.xaxis.type == "category"
    assert list(fig.data[0].x) == ["0,10", "0,30"]
