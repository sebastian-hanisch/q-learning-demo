"""Unabhängige Orakel für Q-Learning: Übergangsmodell und `step` gegen eine Koordinaten-Neuimplementierung, V*/Q* per LP (scipy HiGHS) statt Value
Iteration, Policy-Wert per Lineargleichungssystem, `train` Schritt für Schritt gegen eine Schleifen-Neuimplementierung auf identischem Zufallsstrom,
TD-Update in Erwartung gleich dem Bellman-Operator (exakte Aufzählung der drei Rutsch-Ausgänge), ε-gierige Verteilung und Konvergenz auf Q* im
deterministischen Raster."""

import numpy as np
import pytest

import ql_agent as A
import ql_evaluation as E
import ql_grid as G
import ql_reference as R

linprog = pytest.importorskip("scipy.optimize").linprog

_MOVES = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
_NAME = {0: "N", 1: "S", 2: "E", 3: "W"}
_SIDES = {"N": ("E", "W"), "S": ("E", "W"), "E": ("N", "S"), "W": ("N", "S")}


def _outcomes(rows, cols, slip, s, a):
    """Unabhängige Beschreibung von (s, a): Liste (Wahrscheinlichkeit, Folgezustand, Reward, terminal)."""
    r, c = divmod(s, cols)
    goal = (rows - 1, cols - 1)
    if (r, c) == goal:
        return [(1.0, s, 0.0, True)]
    cliff = {(rows - 1, k) for k in range(1, cols - 1)}
    m = _NAME[a]
    out = []
    for p, d in ((1 - slip, m), (slip / 2, _SIDES[m][0]), (slip / 2, _SIDES[m][1])):
        nr, nc = r + _MOVES[d][0], c + _MOVES[d][1]
        if not (0 <= nr < rows and 0 <= nc < cols):
            nr, nc = r, c
        if (nr, nc) == goal:
            out.append((p, nr * cols + nc, 10.0, True))
        elif (nr, nc) in cliff:
            out.append((p, (rows - 1) * cols, -100.0, False))
        else:
            out.append((p, nr * cols + nc, -1.0, False))
    return out


class _FixedU:
    def __init__(self, u):
        self.u = u

    def random(self):
        return self.u


def test_model_and_step_match_an_independent_coordinate_implementation():
    for rows, cols, slip in [(3, 4, 0.1), (4, 8, 0.3), (6, 5, 0.02), (3, 12, 0.0)]:
        g = G.Grid(rows, cols, slip, 0.9)
        P, Rw = G.build_model(g)
        goal_s = g.state_of(g.goal)
        for s in range(g.n_states):
            for a in range(4):
                out = _outcomes(rows, cols, slip, s, a)
                Pm, Rm = np.zeros(g.n_states), 0.0
                for p, s2, r, d in out:
                    Pm[goal_s if d else s2] += p
                    Rm += p * r
                assert np.allclose(P[s, a], Pm) and Rw[s, a] == pytest.approx(Rm)
                if s == goal_s:
                    continue
                us = [(0.5 * (1 - slip), 0), (1 - slip / 4, 2)] + ([(1 - slip + slip / 4, 1)] if slip > 0 else [])
                for u, k in us:
                    assert G.step(g, s, a, _FixedU(u)) == out[k][1:]


def _vstar_lp(P, Rw, gamma):
    S, nA = Rw.shape
    rows = []
    for s in range(S):
        for a in range(nA):
            row = gamma * P[s, a].copy()
            row[s] -= 1.0
            rows.append(row)
    res = linprog(np.ones(S), A_ub=np.array(rows), b_ub=-Rw.reshape(-1), bounds=[(None, None)] * S, method="highs")
    return res.x


def test_value_iteration_and_policy_evaluation_match_lp_and_linear_solve():
    rng = np.random.default_rng(0)
    for rows, cols, slip, gamma in [(3, 4, 0.1, 0.9), (4, 8, 0.1, 0.95), (4, 6, 0.3, 0.99), (3, 5, 0.0, 0.8)]:
        g = G.Grid(rows, cols, slip, gamma)
        P, Rw = G.build_model(g)
        V, Q, pi = R.value_iteration(P, Rw, gamma)
        Vlp = _vstar_lp(P, Rw, gamma)
        assert np.max(np.abs(V - Vlp)) < 1e-5
        assert np.max(np.abs(Q - (Rw + gamma * P @ Vlp))) < 1e-5
        idx = np.arange(g.n_states)
        for pol in (pi, G.naive_policy(g), rng.integers(0, 4, g.n_states)):
            exact = np.linalg.solve(np.eye(g.n_states) - gamma * P[idx, pol], Rw[idx, pol])
            assert np.max(np.abs(exact - R.policy_evaluation(P, Rw, pol, gamma))) < 1e-4 * max(1.0, np.max(np.abs(exact)))


def _replay(rows, cols, slip, gamma, alpha, eps0, decay, episodes, seed):
    rng = np.random.default_rng(seed)
    Q = [[0.0] * 4 for _ in range(rows * cols)]
    rets, lens = [], []
    for e in range(episodes):
        eps = eps0 if decay <= 0 else max(0.01, eps0 / (1 + decay * e))
        s, tot, n = (rows - 1) * cols, 0.0, 0
        for _ in range(400):
            if rng.random() < eps:
                a = int(rng.integers(4))
            else:
                best = [k for k in range(4) if Q[s][k] == max(Q[s])]
                a = best[0] if len(best) == 1 else int(rng.choice(np.array(best)))
            u = rng.random()
            k = 0 if (slip == 0 or u < 1 - slip) else (1 if u < 1 - slip / 2 else 2)
            _, s2, r, d = _outcomes(rows, cols, slip, s, a)[k]
            Q[s][a] += alpha * ((r if d else r + gamma * max(Q[s2])) - Q[s][a])
            tot, n, s = tot + r, n + 1, s2
            if d:
                break
        rets.append(tot)
        lens.append(n)
    return np.array(Q), np.array(rets), np.array(lens)


def test_train_replays_step_by_step_on_the_same_random_stream():
    rng = np.random.default_rng(1)
    for _ in range(25):
        rows, cols = int(rng.integers(3, 6)), int(rng.integers(4, 8))
        slip, gamma, alpha = float(rng.choice([0, 0.1, 0.3])), float(rng.choice([0.8, 0.95])), float(rng.choice([0.05, 0.5]))
        eps0, decay, ep, seed = float(rng.choice([0.2, 1.0])), float(rng.choice([0, 0.02])), int(rng.integers(2, 30)), int(rng.integers(1000))
        Q, rets, lens, snaps = A.train(G.Grid(rows, cols, slip, gamma), alpha, eps0, decay, ep, seed, checkpoints=(1, ep))
        Qo, ro, lo = _replay(rows, cols, slip, gamma, alpha, eps0, decay, ep, seed)
        assert np.allclose(Q, Qo, atol=1e-12, rtol=0)
        assert np.array_equal(rets, ro) and np.array_equal(lens, lo)
        assert np.array_equal(snaps[ep], Q)


def test_td_update_is_in_expectation_the_bellman_backup_of_the_current_estimate():
    rng = np.random.default_rng(2)
    for _ in range(40):
        rows, cols = int(rng.integers(3, 6)), int(rng.integers(4, 8))
        slip, gamma, alpha = float(rng.choice([0.02, 0.1, 0.3])), float(rng.choice([0.8, 0.95])), float(rng.choice([0.1, 0.5]))
        g = G.Grid(rows, cols, slip, gamma)
        P, Rw = G.build_model(g)
        Q0 = rng.normal(size=(g.n_states, 4)) * 5
        s, a = int(rng.integers(g.n_states)), int(rng.integers(4))
        if s == g.state_of(g.goal):
            continue
        exp = 0.0
        for p, u in ((1 - slip, 0.5 * (1 - slip)), (slip / 2, 1 - slip + slip / 4), (slip / 2, 1 - slip / 4)):
            Q = Q0.copy()
            s2, r, d = G.step(g, s, a, _FixedU(u))
            A.update(Q, s, a, r, s2, d, alpha, gamma)
            exp += p * Q[s, a]
        V = Q0.max(axis=1)
        V[g.state_of(g.goal)] = 0.0                                                                  # terminal: kein Bootstrap
        assert exp == pytest.approx(Q0[s, a] + alpha * (Rw[s, a] + gamma * P[s, a] @ V - Q0[s, a]), abs=1e-9)


def test_epsilon_greedy_action_distribution_and_schedule_follow_the_formula():
    for i, (eps, row) in enumerate([(0.0, [1, 1, 0, 0]), (0.2, [0, 0, 0, 0]), (0.5, [3, 1, 2, 0]), (1.0, [5, 0, 0, 0])]):
        Q = np.array([row], dtype=float)
        best = np.flatnonzero(Q[0] == Q[0].max())
        want = np.full(4, eps / 4)
        want[best] += (1 - eps) / len(best)
        rng = np.random.default_rng(i)
        got = np.bincount([A.choose_action(Q, 0, eps, rng) for _ in range(8000)], minlength=4) / 8000
        assert np.max(np.abs(got - want)) < 0.03
    for e in range(0, 3000, 11):
        assert A.epsilon_by_episode(e, 0.7, 0.01) == max(0.01, 0.7 / (1 + 0.01 * e))


def test_deterministic_grid_converges_exactly_to_qstar_on_all_visited_pairs():
    g = G.Grid(3, 4, 0.0, 0.9)
    P, Rw = G.build_model(g)
    Qstar = Rw + 0.9 * P @ _vstar_lp(P, Rw, 0.9)
    Q, _, _, _ = A.train(g, 0.5, 1.0, 0.0, 600, 0)
    mask = np.ones_like(Q, bool)
    mask[g.state_of(g.goal)] = False
    for rc in g.cliff:
        mask[g.state_of(rc)] = False                                                                 # Klippenzellen werden nie betreten
    assert np.max(np.abs(Q[mask] - Qstar[mask])) < 1e-3


def test_analysis_gap_and_step_counter_against_independent_references():
    for seed in range(4):
        s = E.Settings(rows=3, cols=5, slip=0.1, gamma=0.95, alpha=0.1, epsilon_start=1.0, epsilon_decay=0.01, episodes=40 + seed, seed=seed)
        a = E.analyse(s)
        P, Rw = G.build_model(a.grid)
        idx = np.arange(a.grid.n_states)
        Vpi = np.linalg.solve(np.eye(a.grid.n_states) - a.grid.gamma * P[idx, a.policy], Rw[idx, a.policy])
        s0 = a.grid.state_of(a.grid.start)
        assert a.gap == pytest.approx(_vstar_lp(P, Rw, a.grid.gamma)[s0] - Vpi[s0], abs=1e-4)
        _, _, lens = _replay(3, 5, 0.1, 0.95, 0.1, 1.0, 0.01, s.episodes, seed)
        assert a.env_steps == int(lens.sum())
