"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Trainingsstand-Slider, Permalink-Grenzen/-Raster, Extremwerte, drei Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ql_constants as C
import ql_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info) + list(at.error):
        assert "{de(" not in el.value and "{pct(" not in el.value, el.value[:120]


def test_default_run_shows_metrics_charts_and_a_verdict():
    at = _run()
    _ok(at)
    assert len(at.metric) == 4 and len(at.get("plotly_chart")) >= 2
    assert len(at.success) + len(at.warning) + len(at.error) >= 1


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == pytest.approx(p[key]) if isinstance(p[key], float) else at.session_state[state_key] == p[key]


def test_frame_slider_survives_a_smaller_episode_count():
    at = _run(ql_frame_idx=5)
    _ok(at)
    at.slider(key="episodes_slider").set_value(C.EPISODES_MIN).run()
    _ok(at)


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["rows"] = "999"
    at.query_params["cols"] = "-5"
    at.query_params["slip"] = "abc"
    at.query_params["gamma"] = "5"
    at.query_params["alpha"] = "-1"
    at.query_params["episodes"] = "999999"
    at.run()
    _ok(at)
    s = at.session_state
    assert s["rows_slider"] == C.ROWS_MAX and s["cols_slider"] == C.COLS_MIN
    assert s["slip_slider"] == C.DEFAULT_SLIP and s["gamma_slider"] == C.GAMMA_MAX
    assert s["alpha_slider"] == C.ALPHA_MIN and s["episodes_slider"] == C.EPISODES_MAX


@pytest.mark.parametrize("kw", [
    dict(rows_slider=C.ROWS_MIN, cols_slider=C.COLS_MIN, episodes_slider=C.EPISODES_MIN),
    dict(rows_slider=C.ROWS_MAX, cols_slider=C.COLS_MAX, episodes_slider=C.EPISODES_MIN),
    dict(alpha_slider=C.ALPHA_MIN, epsilon_decay_slider=C.EPSILON_DECAY_MIN),
    dict(alpha_slider=C.ALPHA_MAX, epsilon_decay_slider=C.EPSILON_DECAY_MAX, epsilon_start_slider=C.EPSILON_START_MIN),
])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()
    _ok(at)


def test_episodes_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_EPISODE_CHECKPOINTS", (50, 100))
    monkeypatch.setattr(C, "EXP_SEEDS", 5)
    at = _run()
    _click(at, "episodes_start")
    assert at.session_state["episodes_on"] and any("Befund" in w.value for w in at.warning)


def test_decay_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_EPSILON_DECAY_LEVELS", (0.0, 0.01))
    monkeypatch.setattr(C, "EXP_SEEDS", 5)
    at = _run()
    _click(at, "decay_start")
    assert at.session_state["decay_on"] and any("Zerfall" in w.value for w in at.warning)


def test_alpha_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_ALPHA_LEVELS", (0.10, 0.30))
    monkeypatch.setattr(C, "EXP_SEEDS", 5)
    at = _run()
    _click(at, "alpha_start")
    assert at.session_state["alpha_on"] and any("Lernrate" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
