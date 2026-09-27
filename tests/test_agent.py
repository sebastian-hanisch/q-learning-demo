"""Q-Learning von Hand: die TD-Update-Formel, das GLIE-Epsilon-Schema, gierige/erkundende Aktionswahl mit Gleichstand-Aufloesung, und die
Trainingsschleife gegen eine unabhaengige Schritt-fuer-Schritt-Nachrechnung sowie gegen einen strukturellen Kuerzungstest (Snapshot = kuerzerer Lauf)."""

import numpy as np
import pytest

import ql_agent as A
import ql_constants as C
from ql_grid import Grid, step


def test_epsilon_by_episode_constant_when_decay_is_zero():
    assert A.epsilon_by_episode(0, start=1.0, decay=0.0) == pytest.approx(1.0)
    assert A.epsilon_by_episode(9999, start=1.0, decay=0.0) == pytest.approx(1.0)


def test_epsilon_by_episode_decays_monotonically_towards_but_never_below_eps_min():
    values = [A.epsilon_by_episode(e, start=1.0, decay=0.01) for e in range(0, 2000, 50)]
    assert all(a >= b for a, b in zip(values, values[1:]))                                          # monoton fallend
    assert all(v >= C.EPSILON_MIN for v in values)
    assert values[0] == pytest.approx(1.0)
    assert A.epsilon_by_episode(100, start=1.0, decay=0.01) == pytest.approx(max(C.EPSILON_MIN, 1.0 / (1.0 + 0.01 * 100)))


def test_choose_action_epsilon_zero_is_always_greedy():
    Q = np.zeros((1, 4))
    Q[0] = [1.0, 5.0, 2.0, 0.0]
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert A.choose_action(Q, 0, epsilon=0.0, rng=rng) == 1


def test_choose_action_breaks_ties_among_all_maxima():
    Q = np.zeros((1, 4))
    Q[0] = [5.0, 0.0, 5.0, 0.0]
    rng = np.random.default_rng(1)
    seen = {A.choose_action(Q, 0, epsilon=0.0, rng=rng) for _ in range(200)}
    assert seen == {0, 2}                                                                            # beide Gleichstands-Aktionen kommen vor, nie 1 oder 3


def test_choose_action_epsilon_one_explores_all_actions_roughly_uniformly():
    Q = np.zeros((1, 4))
    Q[0] = [5.0, 0.0, 0.0, 0.0]                                                                      # Aktion 0 waere klar bevorzugt, wird bei epsilon=1 ignoriert
    rng = np.random.default_rng(2)
    counts = np.zeros(4, dtype=int)
    n = 4000
    for _ in range(n):
        counts[A.choose_action(Q, 0, epsilon=1.0, rng=rng)] += 1
    assert np.all(np.abs(counts / n - 0.25) < 0.03)


def test_update_td_target_for_a_non_terminal_transition_by_hand():
    Q = np.array([[1.0, 2.0], [3.0, 4.0]])
    A.update(Q, s=0, a=0, r=1.0, s_next=1, done=False, alpha=0.5, gamma=0.9)
    target = 1.0 + 0.9 * 4.0                                                                          # max(Q[1]) = 4.0
    assert Q[0, 0] == pytest.approx(1.0 + 0.5 * (target - 1.0))


def test_update_td_target_for_a_terminal_transition_ignores_the_bootstrap():
    Q = np.array([[1.0, 2.0], [100.0, 100.0]])                                                        # absichtlich hohe Werte im Folgezustand
    A.update(Q, s=0, a=0, r=10.0, s_next=1, done=True, alpha=0.5, gamma=0.9)
    assert Q[0, 0] == pytest.approx(1.0 + 0.5 * (10.0 - 1.0))                                          # kein gamma*max(Q[1]) im Ziel


def test_run_episode_reproduces_a_hand_composed_step_by_step_replay():
    # Deterministisches Raster (kein Rutschen) auf einem winzigen 1x3-Gitter: die einzige verbleibende Zufaelligkeit ist die
    # epsilon-gierige Aktionswahl. Reproduziert dieselbe RNG-Ziehungsreihenfolge wie run_episode, Schritt fuer Schritt von Hand.
    grid = Grid(rows=1, cols=3, slip=0.0, gamma=0.9)
    rng_a = np.random.default_rng(7)
    Q_a = np.zeros((grid.n_states, A.N_ACTIONS))
    total_a, steps_a = A.run_episode(grid, Q_a, epsilon=0.3, alpha=0.2, rng=rng_a, max_steps=20)

    rng_b = np.random.default_rng(7)
    Q_b = np.zeros((grid.n_states, A.N_ACTIONS))
    s = grid.state_of(grid.start)
    total_b, steps_b = 0.0, 0
    for _ in range(20):
        a = A.choose_action(Q_b, s, 0.3, rng_b)
        s_next, r, done = step(grid, s, a, rng_b)
        A.update(Q_b, s, a, r, s_next, done, alpha=0.2, gamma=grid.gamma)
        total_b += r
        steps_b += 1
        s = s_next
        if done:
            break
    assert steps_a == steps_b and total_a == pytest.approx(total_b)
    assert np.array_equal(Q_a, Q_b)


def test_train_snapshot_at_a_checkpoint_equals_a_separately_truncated_run():
    # Trainiert man 1500 Episoden und schaut sich den Snapshot nach 100 an, muss das exakt dasselbe Ergebnis liefern wie ein eigener
    # Lauf mit nur 100 Episoden (gleicher Seed) - dieselbe RNG-Konsumreihenfolge, kein Seiteneffekt aus den Episoden danach.
    grid = Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    _, _, _, snapshots = A.train(grid, alpha=0.1, epsilon_start=1.0, epsilon_decay=0.005, episodes=1500, seed=3, checkpoints=(100,))
    Q_short, _, _, _ = A.train(grid, alpha=0.1, epsilon_start=1.0, epsilon_decay=0.005, episodes=100, seed=3)
    assert np.array_equal(snapshots[100], Q_short)


def test_train_returns_arrays_of_the_right_shape():
    grid = Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    Q, returns, lengths, snapshots = A.train(grid, alpha=0.1, epsilon_start=1.0, epsilon_decay=0.005, episodes=50, seed=0)
    assert Q.shape == (grid.n_states, A.N_ACTIONS)
    assert returns.shape == (50,) and lengths.shape == (50,)
    assert np.all(lengths >= 1) and np.all(lengths <= C.MAX_STEPS_PER_EPISODE)
