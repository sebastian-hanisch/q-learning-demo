"""SETTING_SPECS-Permalink-Muster, Presets und Regler-Grenzen (Standardmuster des Portfolios)."""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import ql_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "rows_slider": SettingSpec("rows", int, C.DEFAULT_ROWS, C.ROWS_MIN, C.ROWS_MAX),
    "cols_slider": SettingSpec("cols", int, C.DEFAULT_COLS, C.COLS_MIN, C.COLS_MAX),
    "slip_slider": SettingSpec("slip", float, C.DEFAULT_SLIP, C.SLIP_MIN, C.SLIP_MAX),
    "gamma_slider": SettingSpec("gamma", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX),
    "alpha_slider": SettingSpec("alpha", float, C.DEFAULT_ALPHA, C.ALPHA_MIN, C.ALPHA_MAX),
    "epsilon_start_slider": SettingSpec("eps0", float, C.DEFAULT_EPSILON_START, C.EPSILON_START_MIN, C.EPSILON_START_MAX),
    "epsilon_decay_slider": SettingSpec("decay", float, C.DEFAULT_EPSILON_DECAY, C.EPSILON_DECAY_MIN, C.EPSILON_DECAY_MAX),
    "episodes_slider": SettingSpec("episodes", int, C.DEFAULT_EPISODES, C.EPISODES_MIN, C.EPISODES_MAX),
}
PRESET_KEYS = {
    "rows": "rows_slider", "cols": "cols_slider", "slip": "slip_slider", "gamma": "gamma_slider",
    "alpha": "alpha_slider", "epsilon_start": "epsilon_start_slider", "epsilon_decay": "epsilon_decay_slider", "episodes": "episodes_slider",
}
STEPS = {
    "slip_slider": C.SLIP_STEP, "gamma_slider": C.GAMMA_STEP, "alpha_slider": C.ALPHA_STEP,
    "epsilon_start_slider": C.EPSILON_START_STEP, "epsilon_decay_slider": C.EPSILON_DECAY_STEP, "episodes_slider": C.EPISODES_STEP,
}


def _p(**kw):
    base = {
        "rows": C.DEFAULT_ROWS, "cols": C.DEFAULT_COLS, "slip": C.DEFAULT_SLIP, "gamma": C.DEFAULT_GAMMA,
        "alpha": C.DEFAULT_ALPHA, "epsilon_start": C.DEFAULT_EPSILON_START, "epsilon_decay": C.DEFAULT_EPSILON_DECAY, "episodes": C.DEFAULT_EPISODES,
    }
    base.update(kw)
    return base


PRESETS = {
    "Standardfall": _p(),
    "Zu wenig Training": _p(episodes=200),
    "Konstant explorativ (kein GLIE)": _p(epsilon_decay=0.0),
    "Hohe Lernrate": _p(alpha=0.50),
    "Kein Rutschen (deterministisch)": _p(slip=0.0),
    "Großes Raster": _p(rows=6, cols=12, episodes=3000),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = round(float(snapped), 4)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


PRESET_HELP = {
    "Standardfall": "4×8-Raster, Rutschen 0,10, α=0,10, Epsilon-Zerfall 0,005, 1500 Episoden: guter, aber nicht garantierter Kompromiss - nur etwa 40 % der Läufe (8 von 20 Seeds) landen nahe am Optimum, der Rest bleibt spürbar schlechter.",
    "Zu wenig Training": "Nur 200 Episoden: die Q-Schätzung ist noch verrauscht - eine harmlos aussehende Gewohnheit (z. B. am Start gegen die Wand laufen) kann dabei genauso billig wirken wie die echte Optimalroute weg von der Klippe.",
    "Konstant explorativ (kein GLIE)": "Epsilon bleibt bei 1,0 (rein zufälliges Verhalten während des ganzen Trainings): Q-Learning ist off-policy und lernt trotzdem, aber deutlich langsamer, weil die Erfahrung nie auf die produktive Gegend um den optimalen Weg konzentriert wird.",
    "Hohe Lernrate": "α=0,50: jede neue Erfahrung überschreibt die Q-Schätzung fast vollständig statt sie zu mitteln - in seltenen Fällen wird dabei eine Zelle direkt über der Klippe zur \"gierigen\" Wahl, mit sehr teuren Ausreißern.",
    "Kein Rutschen (deterministisch)": "Ohne Rutschen ist der direkte Weg entlang der Klippe optimal (wie in value-iteration-demo) - für den Agenten ändert sich nur die Umgebung, nicht der Lernalgorithmus.",
    "Großes Raster": "6×12-Raster (72 States, 3000 Episoden): mehr States bedeuten mehr Tabelleneinträge, die alle besucht werden müssen, bevor die Policy überall verlässlich ist.",
}
