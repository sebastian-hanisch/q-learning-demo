"""Die Referenzloesung (Value Iteration, nur zur Gegenprobe): die Bellman-Formel von Hand und ein entarteter Fall mit geschlossener Loesung -
dieselben Pruefungen wie in `value-iteration-demo`, hier nur als Massstab fuer Q-Learning gebraucht."""

import numpy as np
import pytest

import ql_grid as G
import ql_reference as R


def test_q_values_by_hand():
    P = np.array([[[0.5, 0.5], [1.0, 0.0]]])
    Rw = np.array([[1.0, 2.0]])
    V = np.array([10.0, 20.0])
    Q = R.q_values(P, Rw, V, gamma=0.9)
    assert Q[0, 0] == pytest.approx(1.0 + 0.9 * (0.5 * 10 + 0.5 * 20)) and Q[0, 1] == pytest.approx(2.0 + 0.9 * 10.0)


def test_degenerate_single_row_grid_gives_up_and_bounces_forever():
    g = G.Grid(rows=1, cols=3, slip=0.0, gamma=0.95)
    P, Rw = G.build_model(g)
    V, Q, policy = R.value_iteration(P, Rw, g.gamma)
    s0 = g.state_of(g.start)
    assert V[s0] == pytest.approx(-1.0 / (1.0 - g.gamma), abs=1e-4)
    assert policy[s0] == G.NORTH


def test_policy_evaluation_of_a_fixed_policy_by_hand():
    P = np.array([[[1.0]]])
    Rw = np.array([[1.0]])
    V = R.policy_evaluation(P, Rw, np.array([0]), gamma=0.5)
    assert V[0] == pytest.approx(2.0, abs=1e-6)


def test_policy_evaluation_of_the_optimal_policy_matches_value_iteration():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    P, Rw = G.build_model(g)
    V_star, _, pi_star = R.value_iteration(P, Rw, g.gamma)
    V_pi = R.policy_evaluation(P, Rw, pi_star, g.gamma)
    assert np.allclose(V_star, V_pi, atol=1e-3)


def test_a_worse_policy_has_a_lower_value_at_the_start():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    P, Rw = G.build_model(g)
    V_star, _, pi_star = R.value_iteration(P, Rw, g.gamma)
    naive = G.naive_policy(g)
    V_naive = R.policy_evaluation(P, Rw, naive, g.gamma)
    s0 = g.state_of(g.start)
    assert V_naive[s0] <= V_star[s0] + 1e-6
