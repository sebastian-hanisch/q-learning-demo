"""Auswertung: Q-Learning gegen die Value-Iteration-Referenz auf demselben Modell, dazu drei Experimente (Konvergenz über die Trainingsdauer, Wirkung
des Epsilon-Zerfalls [GLIE gegen konstant explorativ], Wirkung der Lernrate alpha). Ergebnisse sind Mittel über Seeds mit Streuung (SE) - Q-Learning
ist stochastisch, anders als die exakte Referenz."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import ql_agent as A
import ql_constants as C
import ql_grid as G
import ql_reference as R


@dataclass(frozen=True)
class Settings:
    rows: int = C.DEFAULT_ROWS
    cols: int = C.DEFAULT_COLS
    slip: float = C.DEFAULT_SLIP
    gamma: float = C.DEFAULT_GAMMA
    alpha: float = C.DEFAULT_ALPHA
    epsilon_start: float = C.DEFAULT_EPSILON_START
    epsilon_decay: float = C.DEFAULT_EPSILON_DECAY
    episodes: int = C.DEFAULT_EPISODES
    seed: int = 0

    @property
    def grid(self):
        return G.Grid(self.rows, self.cols, self.slip, self.gamma)


@lru_cache(maxsize=64)
def _reference(rows, cols, slip, gamma):
    """Die bekannte, exakte Loesung (Value Iteration) - der Massstab, an dem sich der lernende Agent messen lassen muss. Wird ausschliesslich zur
    Auswertung/Gegenprobe verwendet, nie vom Agenten selbst."""
    grid = G.Grid(rows, cols, slip, gamma)
    P, Rw = G.build_model(grid)
    V_star, Q_star, pi_star = R.value_iteration(P, Rw, gamma)
    return grid, P, Rw, V_star, Q_star, pi_star


@dataclass
class Analysis:
    settings: Settings
    grid: G.Grid
    Q: np.ndarray
    returns: np.ndarray
    lengths: np.ndarray
    policy: np.ndarray
    V_star: np.ndarray
    pi_star: np.ndarray
    V_pi: np.ndarray
    gap: float
    env_steps: int
    snapshots: dict


def analyse(s, checkpoints=()):
    grid, P, Rw, V_star, Q_star, pi_star = _reference(s.rows, s.cols, s.slip, s.gamma)
    Q, returns, lengths, snapshots = A.train(grid, s.alpha, s.epsilon_start, s.epsilon_decay, s.episodes, s.seed, checkpoints=checkpoints)
    policy = Q.argmax(axis=1)
    V_pi = R.policy_evaluation(P, Rw, policy, gamma=grid.gamma)
    start_s = grid.state_of(grid.start)
    gap = float(V_star[start_s] - V_pi[start_s])
    return Analysis(s, grid, Q, returns, lengths, policy, V_star, pi_star, V_pi, gap, int(lengths.sum()), snapshots)


def policy_gap(rows, cols, slip, gamma, alpha, epsilon_start, epsilon_decay, episodes, seed):
    """Der Wert-Abstand V*(start) - V_gelernte_Policy(start) fuer einen einzelnen Trainingslauf - die zentrale Kennzahl aller Experimente."""
    grid, P, Rw, V_star, _, _ = _reference(rows, cols, slip, gamma)
    Q, _, lengths, _ = A.train(grid, alpha, epsilon_start, epsilon_decay, episodes, seed)
    policy = Q.argmax(axis=1)
    V_pi = R.policy_evaluation(P, Rw, policy, gamma=gamma)
    start_s = grid.state_of(grid.start)
    return float(V_star[start_s] - V_pi[start_s]), int(lengths.sum())


def _summary(gaps, env_steps):
    gaps = np.asarray(gaps, dtype=float)
    env_steps = np.asarray(env_steps, dtype=float)
    se = float(gaps.std(ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else 0.0
    return {
        "mean": float(gaps.mean()),
        "se": se,
        "median": float(np.median(gaps)),
        "frac_near_optimal": float(np.mean(gaps < C.NEAR_OPTIMAL_GAP)),
        "frac_cliff": float(np.mean(gaps > C.CLIFF_GAP_THRESHOLD)),
        "env_steps_mean": float(env_steps.mean()),
    }


# --- Experiment 1: Konvergenz ueber die Trainingsdauer (Hook 3: mehr Umgebungsschritte als Value Iteration Bellman-Backups) --------------------------

def episodes_experiment(checkpoints=None, seeds=None, base=None):
    checkpoints = C.EXP_EPISODE_CHECKPOINTS if checkpoints is None else checkpoints
    seeds = range(C.EXP_SEEDS) if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for episodes in checkpoints:
        gaps, env_steps = [], []
        for sd in seeds:
            gap, steps = policy_gap(base.rows, base.cols, base.slip, base.gamma, base.alpha, base.epsilon_start, base.epsilon_decay, episodes, sd)
            gaps.append(gap)
            env_steps.append(steps)
        rows.append({"episodes": episodes, **_summary(gaps, env_steps)})
    return {"n_seeds": len(list(range(C.EXP_SEEDS)) if seeds is None else list(seeds)), "checkpoints": tuple(checkpoints), "rows": rows}


# --- Experiment 2: Epsilon-Zerfall (GLIE) gegen konstant explorativ -----------------------------------------------------------------------------------

def decay_experiment(levels=None, seeds=None, base=None):
    levels = C.EXP_EPSILON_DECAY_LEVELS if levels is None else levels
    seeds = range(C.EXP_SEEDS) if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for decay in levels:
        gaps, env_steps = [], []
        for sd in seeds:
            gap, steps = policy_gap(base.rows, base.cols, base.slip, base.gamma, base.alpha, base.epsilon_start, decay, base.episodes, sd)
            gaps.append(gap)
            env_steps.append(steps)
        rows.append({"decay": decay, **_summary(gaps, env_steps)})
    return {"n_seeds": len(list(range(C.EXP_SEEDS)) if seeds is None else list(seeds)), "levels": tuple(levels), "rows": rows}


# --- Experiment 3: Lernrate alpha ------------------------------------------------------------------------------------------------------------------

def alpha_experiment(levels=None, seeds=None, base=None):
    levels = C.EXP_ALPHA_LEVELS if levels is None else levels
    seeds = range(C.EXP_SEEDS) if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for alpha in levels:
        gaps, env_steps = [], []
        for sd in seeds:
            gap, steps = policy_gap(base.rows, base.cols, base.slip, base.gamma, alpha, base.epsilon_start, base.epsilon_decay, base.episodes, sd)
            gaps.append(gap)
            env_steps.append(steps)
        rows.append({"alpha": alpha, **_summary(gaps, env_steps)})
    return {"n_seeds": len(list(range(C.EXP_SEEDS)) if seeds is None else list(seeds)), "levels": tuple(levels), "rows": rows}
