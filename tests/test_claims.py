"""Jede Zahl aus dem README, nachgerechnet ueber die echten Auswertungsfunktionen - nie per Hand abgetippt. Q-Learning ist stochastisch: Zahlen aus
Mehr-Seed-Experimenten tragen grosszuegige Baender (CI-robust, siehe feedback_ci_platform_robust_tests), der Standardfall (fester Seed) ist exakt."""

import pytest

import ql_constants as C
import ql_evaluation as E


def test_v_star_start_and_vi_cost():
    grid, P, R, V_star, _, _ = E._reference(C.DEFAULT_ROWS, C.DEFAULT_COLS, C.DEFAULT_SLIP, C.DEFAULT_GAMMA)
    s0 = grid.state_of(grid.start)
    assert V_star[s0] == pytest.approx(-9.23, abs=0.01)


def test_standard_case():
    a = E.analyse(E.Settings())
    assert a.gap == pytest.approx(2.40, abs=0.02)
    assert a.V_pi[a.grid.state_of(a.grid.start)] == pytest.approx(-11.62, abs=0.02)
    assert a.env_steps == pytest.approx(52917, abs=200)
    reachable = [s for s in range(a.grid.n_states) if a.grid.rc_of(s) not in a.grid.cliff]
    import numpy as np
    match = float(np.mean(a.policy[reachable] == a.pi_star[reachable]))
    assert match == pytest.approx(0.962, abs=0.01)


def test_episodes_experiment():
    ee = E.episodes_experiment()
    rows = {r["episodes"]: r for r in ee["rows"]}
    assert rows[100]["frac_near_optimal"] == pytest.approx(0.50, abs=0.10)
    assert rows[4000]["frac_near_optimal"] == pytest.approx(0.55, abs=0.10)
    assert rows[4000]["env_steps_mean"] == pytest.approx(107147, rel=0.05)
    vi_backups = 44 * 32
    assert rows[4000]["env_steps_mean"] / vi_backups == pytest.approx(76, rel=0.1)


def test_decay_experiment():
    de = E.decay_experiment(base=E.Settings(episodes=C.EXP_EPISODES))
    rows = {r["decay"]: r for r in de["rows"]}
    assert rows[0.0]["frac_near_optimal"] == pytest.approx(0.30, abs=0.10)
    assert rows[0.005]["frac_near_optimal"] == pytest.approx(0.60, abs=0.10)
    assert rows[0.0]["env_steps_mean"] / rows[0.005]["env_steps_mean"] == pytest.approx(7.6, rel=0.2)
    best = min(de["rows"], key=lambda r: r["mean"])
    assert best["decay"] == pytest.approx(0.005)


def test_alpha_experiment():
    al = E.alpha_experiment(base=E.Settings(episodes=C.EXP_EPISODES))
    rows = {r["alpha"]: r for r in al["rows"]}
    assert rows[0.05]["frac_near_optimal"] == pytest.approx(0.65, abs=0.10)
    assert rows[0.50]["frac_near_optimal"] == pytest.approx(0.15, abs=0.10)
    assert rows[0.50]["frac_cliff"] == pytest.approx(0.05, abs=0.05)
