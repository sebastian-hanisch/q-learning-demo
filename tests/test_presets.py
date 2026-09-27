"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import ql_constants as C
import ql_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            assert spec.lo <= p[key] <= spec.hi, (key, p[key])
        for key, state_key in P.PRESET_KEYS.items():
            if state_key not in P.STEPS:
                continue
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-6, (key, p[key])


def test_standard_preset_equals_the_default_settings():
    p = P.PRESETS["Standardfall"]
    assert p["rows"] == C.DEFAULT_ROWS and p["cols"] == C.DEFAULT_COLS
    assert p["alpha"] == C.DEFAULT_ALPHA and p["epsilon_decay"] == C.DEFAULT_EPSILON_DECAY and p["episodes"] == C.DEFAULT_EPISODES


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("slip_slider") == (C.SLIP_MIN, C.SLIP_MAX) and P.bounds("rows_slider") == (C.ROWS_MIN, C.ROWS_MAX)
    assert P.bounds("alpha_slider") == (C.ALPHA_MIN, C.ALPHA_MAX)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
