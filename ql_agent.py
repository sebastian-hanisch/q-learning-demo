"""Q-Learning (Watkins & Dayan 1992): Off-Policy-TD-Kontrolle. Der Agent sieht nur einzelne Übergänge (`ql_grid.step`), nie das Modell (P, R) selbst -
anders als Value Iteration in `value-iteration-demo`, das direkt auf dem bekannten Modell rechnet."""

import numpy as np

import ql_constants as C
from ql_grid import ACTIONS, step

N_ACTIONS = len(ACTIONS)


def epsilon_by_episode(episode, start, decay, eps_min=C.EPSILON_MIN):
    """GLIE-Schema (greedy in the limit with infinite exploration): epsilon faellt monoton gegen `eps_min`, erreicht ihn aber nie exakt - decay=0 haelt
    epsilon konstant bei `start` (dann NICHT GLIE, absichtlich als Kontrastregler)."""
    if decay <= 0.0:
        return start
    return max(eps_min, start / (1.0 + decay * episode))


def choose_action(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    row = Q[state]
    best = np.flatnonzero(row == row.max())
    return int(best[0]) if best.size == 1 else int(rng.choice(best))


def update(Q, s, a, r, s_next, done, alpha, gamma):
    target = r if done else r + gamma * Q[s_next].max()
    Q[s, a] += alpha * (target - Q[s, a])


def run_episode(grid, Q, epsilon, alpha, rng, max_steps=C.MAX_STEPS_PER_EPISODE):
    s = grid.state_of(grid.start)
    total_reward, steps = 0.0, 0
    for _ in range(max_steps):
        a = choose_action(Q, s, epsilon, rng)
        s_next, r, done = step(grid, s, a, rng)
        update(Q, s, a, r, s_next, done, alpha, grid.gamma)
        total_reward += r
        steps += 1
        s = s_next
        if done:
            break
    return total_reward, steps


def train(grid, alpha, epsilon_start, epsilon_decay, episodes, seed, max_steps=C.MAX_STEPS_PER_EPISODE, checkpoints=()):
    """Trainiert `episodes` Episoden, gibt die gelernte Q-Tabelle sowie je Episode Ertrag/Schrittzahl zurueck. `checkpoints` (Episodenzahlen) liefern
    zusaetzlich eine Kopie der Q-Tabelle NACH genau dieser vielen Episoden - fuer die Episode-fuer-Episode-Ansicht, ohne fuer jeden Slider-Wert neu
    trainieren zu muessen."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((grid.n_states, N_ACTIONS))
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    snapshots = {}
    checkpoint_set = set(checkpoints)
    for e in range(episodes):
        epsilon = epsilon_by_episode(e, epsilon_start, epsilon_decay)
        total_reward, steps = run_episode(grid, Q, epsilon, alpha, rng, max_steps)
        returns[e] = total_reward
        lengths[e] = steps
        if (e + 1) in checkpoint_set:
            snapshots[e + 1] = Q.copy()
    return Q, returns, lengths, snapshots
