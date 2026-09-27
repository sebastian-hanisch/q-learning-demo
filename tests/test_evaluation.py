"""Analyse und die drei Experimente: Aufbau, und die Korrektheits-Kette dieses Stuecks - Q-Learnings gelernte gierige Policy muss nach ausreichendem
Training auf demselben (fuer den Test bekannten) MDP nahe an Value Iterations Optimum liegen. Exakte Zahlen stehen in test_claims.py."""

import numpy as np
import pytest

import ql_evaluation as E


@pytest.fixture(scope="module")
def analysis():
    return E.analyse(E.Settings(episodes=300, seed=0))


def test_analyse_wiring(analysis):
    a = analysis
    assert a.Q.shape == (a.grid.n_states, 4)
    assert a.returns.shape == (300,) and a.lengths.shape == (300,)
    assert a.env_steps == int(a.lengths.sum())
    assert a.gap == pytest.approx(float(a.V_star[a.grid.state_of(a.grid.start)] - a.V_pi[a.grid.state_of(a.grid.start)]))


def test_analyse_is_deterministic_given_the_same_seed():
    s = E.Settings(rows=3, cols=4, episodes=200, seed=5)
    a1, a2 = E.analyse(s), E.analyse(s)
    assert np.array_equal(a1.Q, a2.Q) and a1.gap == pytest.approx(a2.gap)


def test_reference_is_cached_across_settings_with_the_same_grid():
    assert E._reference(4, 8, 0.10, 0.95) is E._reference(4, 8, 0.10, 0.95)


def test_q_learning_reaches_a_near_optimal_policy_on_a_small_grid_across_most_seeds():
    # Kleines Raster (12 States), damit der Test schnell bleibt. Gemessen (Vormessung, 25 Seeds, 600 Episoden): median gap ~ 0,
    # ueber 80 % der Laeufe innerhalb 0,5 vom Optimum - grosszuegige Schwellen wegen der Streuung einzelner Laeufe (CI-robust).
    gaps = [E.policy_gap(3, 4, 0.10, 0.95, alpha=0.10, epsilon_start=1.0, epsilon_decay=0.01, episodes=600, seed=sd)[0] for sd in range(25)]
    gaps = np.array(gaps)
    assert np.median(gaps) < 0.5
    assert np.mean(gaps < 0.5) > 0.6


def test_more_training_needs_more_environment_steps_than_value_iteration_bellman_backups():
    # Hook 3: Q-Learning braucht sichtbar MEHR Umgebungsschritte als Value Iterations Bellman-Backups (Sweeps * n_states) auf demselben Raster.
    grid, P, Rw, V_star, _, pi_star = E._reference(4, 8, 0.10, 0.95)
    from ql_reference import value_iteration
    _, _, _ = value_iteration(P, Rw, grid.gamma)
    # 44 Sweeps * 32 States = 1408 Bellman-Backups (siehe value-iteration-demo, Standardfall) - Q-Learning braucht ein Vielfaches an Umgebungsschritten.
    gap, steps = E.policy_gap(4, 8, 0.10, 0.95, alpha=0.10, epsilon_start=1.0, epsilon_decay=0.005, episodes=1500, seed=0)
    assert steps > 10 * 1408


def test_episodes_experiment_shape():
    exp = E.episodes_experiment(checkpoints=(50, 100), seeds=range(5))
    assert len(exp["rows"]) == 2
    for row in exp["rows"]:
        assert "mean" in row and "median" in row and "frac_near_optimal" in row and "env_steps_mean" in row


def test_decay_experiment_shape():
    exp = E.decay_experiment(levels=(0.0, 0.01), seeds=range(5), base=E.Settings(episodes=300))
    assert len(exp["rows"]) == 2


def test_alpha_experiment_shape():
    exp = E.alpha_experiment(levels=(0.10, 0.30), seeds=range(5), base=E.Settings(episodes=300))
    assert len(exp["rows"]) == 2
