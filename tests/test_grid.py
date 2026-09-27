"""Das Vehikel von Hand nachgerechnet: Zustände, Zellenarten, das Übergangsmodell (fuer die Referenzloesung) UND die gesampelte Einzelschritt-Funktion
`step` (das Einzige, was der lernende Agent von der Umgebung sieht)."""

import numpy as np
import pytest

import ql_grid as G


class _FakeRNG:
    """Liefert `random()` genau die vorgegebenen Werte der Reihe nach - fuer deterministische Zweigwahl in `step`."""

    def __init__(self, values):
        self._values = iter(values)

    def random(self):
        return next(self._values)


def test_start_goal_and_cliff_positions():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    assert g.start == (3, 0) and g.goal == (3, 7)
    assert g.cliff == frozenset((3, c) for c in range(1, 7))
    assert g.n_states == 32


def test_cell_kind_by_hand():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    assert g.cell_kind(g.start) == "start" and g.cell_kind(g.goal) == "goal"
    assert g.cell_kind((3, 3)) == "cliff" and g.cell_kind((0, 0)) == "free"


def test_state_index_round_trip():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    for r in range(g.rows):
        for c in range(g.cols):
            assert g.rc_of(g.state_of((r, c))) == (r, c)


def test_transition_probabilities_sum_to_one_everywhere():
    g = G.Grid(rows=4, cols=8, slip=0.15, gamma=0.95)
    P, R = G.build_model(g)
    assert np.allclose(P.sum(axis=2), 1.0)


def test_goal_is_absorbing_with_zero_reward_in_the_model():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    P, R = G.build_model(g)
    goal_s = g.state_of(g.goal)
    for a in G.ACTIONS:
        assert P[goal_s, a, goal_s] == pytest.approx(1.0) and R[goal_s, a] == pytest.approx(0.0)


def test_entering_the_cliff_in_the_model_costs_the_penalty_and_resets_to_start():
    g = G.Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    P, R = G.build_model(g)
    s = g.state_of((2, 3))
    start_s = g.state_of(g.start)
    assert P[s, G.SOUTH, start_s] == pytest.approx(1.0) and R[s, G.SOUTH] == pytest.approx(-100.0)


def test_naive_policy_walks_the_row_above_the_cliff_then_south():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    pol = G.naive_policy(g)
    row = g.rows - 2
    for c in range(g.cols - 1):
        assert pol[g.state_of((row, c))] == G.EAST
    assert pol[g.state_of((row, g.cols - 1))] == G.SOUTH


# --- step(): das einzige, was der lernende Agent sieht ------------------------------------------------------------------------------------------

def test_step_deterministic_move_no_slip():
    g = G.Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    s = g.state_of((0, 3))
    ns, r, done = G.step(g, s, G.EAST, _FakeRNG([0.5]))
    assert ns == g.state_of((0, 4)) and r == pytest.approx(-1.0) and done is False


def test_step_boundary_bounce_back():
    g = G.Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    s = g.state_of((0, 0))
    ns, r, done = G.step(g, s, G.NORTH, _FakeRNG([0.5]))
    assert ns == s and r == pytest.approx(-1.0) and done is False


def test_step_slip_thresholds_by_hand():
    # slip=0.2: u < 0.8 -> Absicht (Osten), 0.8 <= u < 0.9 -> Norden, u >= 0.9 -> Sueden (die beiden Richtungen quer zu Ost/West).
    g = G.Grid(rows=4, cols=8, slip=0.2, gamma=0.95)
    s = g.state_of((1, 3))
    ns_intended, _, _ = G.step(g, s, G.EAST, _FakeRNG([0.5]))
    ns_ortho0, _, _ = G.step(g, s, G.EAST, _FakeRNG([0.85]))
    ns_ortho1, _, _ = G.step(g, s, G.EAST, _FakeRNG([0.95]))
    assert ns_intended == g.state_of((1, 4))
    assert {ns_ortho0, ns_ortho1} == {g.state_of((0, 3)), g.state_of((2, 3))}                       # Norden und Sueden, in irgendeiner Zuordnung


def test_step_entering_the_cliff_resets_to_start_and_does_not_end_the_episode():
    g = G.Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    s = g.state_of((2, 3))
    ns, r, done = G.step(g, s, G.SOUTH, _FakeRNG([0.5]))
    assert ns == g.state_of(g.start) and r == pytest.approx(-100.0) and done is False


def test_step_entering_the_goal_ends_the_episode():
    g = G.Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    s = g.state_of((2, 7))
    ns, r, done = G.step(g, s, G.SOUTH, _FakeRNG([0.5]))
    assert ns == g.state_of(g.goal) and r == pytest.approx(10.0) and done is True


def test_step_from_the_goal_stays_there_with_zero_reward():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    s = g.state_of(g.goal)
    ns, r, done = G.step(g, s, G.NORTH, _FakeRNG([0.01]))
    assert ns == s and r == pytest.approx(0.0) and done is True


def test_step_matches_the_models_transition_probabilities_statistically():
    # Ueber viele Samples muss die Haeufigkeit jeder Folgezelle der Modell-Wahrscheinlichkeit P[s,a,:] entsprechen (grosszuegiges Band).
    g = G.Grid(rows=4, cols=8, slip=0.2, gamma=0.95)
    P, _ = G.build_model(g)
    s = g.state_of((1, 3))
    rng = np.random.default_rng(0)
    n = 20000
    counts = {}
    for _ in range(n):
        ns, _, _ = G.step(g, s, G.EAST, rng)
        counts[ns] = counts.get(ns, 0) + 1
    for ns, p in enumerate(P[s, G.EAST]):
        if p > 0:
            assert abs(counts.get(ns, 0) / n - p) < 0.02
        else:
            assert counts.get(ns, 0) == 0
